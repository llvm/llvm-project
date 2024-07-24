//===------- cc1modbuildd_main.cpp - Clang CC1 Module Build Daemon --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/ModuleBuildDaemon/SocketSupport.h"
#include "clang/Tooling/ModuleBuildDaemon/Utils.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/ThreadPool.h"

#include <csignal>
#include <cstdbool>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>
#include <system_error>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

using namespace clang::tooling::cc1modbuildd;

// Create unbuffered STDOUT stream so that any logging done by the module build
// daemon can be viewed without having to terminate the process
static llvm::raw_fd_ostream &unbuff_outs() {
  static llvm::raw_fd_ostream S(fileno(stdout), false, true);
  return S;
}

static bool LogVerbose = false;
static void logVerbose(const llvm::Twine &message) {
  if (LogVerbose) {
    unbuff_outs() << message << '\n';
  }
}

static void modifySignals(decltype(SIG_DFL) handler) {
  if (std::signal(SIGTERM, handler) == SIG_ERR) {
    llvm::errs() << "failed to handle SIGTERM" << '\n';
    exit(EXIT_FAILURE);
  }
  if (std::signal(SIGINT, handler) == SIG_ERR) {
    llvm::errs() << "failed to handle SIGINT" << '\n';
    exit(EXIT_FAILURE);
  }
#ifdef SIGHUP
  if (::signal(SIGHUP, SIG_IGN) == SIG_ERR) {
    llvm::errs() << "failed to handle SIGHUP" << '\n';
    exit(EXIT_FAILURE);
  }
#endif
}

namespace {

class ModuleBuildDaemonServer {
public:
  llvm::SmallString<256> SocketPath;
  llvm::SmallString<256> Stderr; // path to stderr
  llvm::SmallString<256> Stdout; // path to stdout

  explicit ModuleBuildDaemonServer(llvm::StringRef Path)
      : SocketPath(Path), Stderr(Path), Stdout(Path) {
    llvm::sys::path::append(SocketPath, SocketFileName);
    llvm::sys::path::append(Stdout, StdoutFileName);
    llvm::sys::path::append(Stderr, StderrFileName);
  }

  void setupDaemonEnv();
  void createDaemonSocket();
  void listenForClients();

  static void
  handleConnection(std::shared_ptr<llvm::raw_socket_stream> Connection);

  // TODO: modify so when shutdownDaemon is called the daemon stops accepting
  // new client connections and waits for all existing client connections to
  // terminate before closing the file descriptor and exiting
  // Meant to be called by signal handler to clean up resources
  void shutdownDaemon() {
    RunServiceLoop = false;
    // Signal handler is installed after ServerListener is created and emplaced
    // into the std::optional<llvm::ListeningSocket>
    ServerListener.value().shutdown();
  }

private:
  std::atomic<bool> RunServiceLoop = true;
  // llvm::ListeningSocket does not have a default constructor so use
  // std::optional as storage
  std::optional<llvm::ListeningSocket> ServerListener;
};

// Used to handle signals
ModuleBuildDaemonServer *DaemonPtr = nullptr;
// DaemonPtr is set to a valid ModuleBuildDaemonServer before the signal handler
// is installed so there is no need to check if DaemonPtr equals nullptr
void handleSignal(int) { DaemonPtr->shutdownDaemon(); }
} // namespace

// Sets up file descriptors and signals for module build daemon
void ModuleBuildDaemonServer::setupDaemonEnv() {
#ifdef _WIN32
  if (std::freopen("NUL", "r", stdin) == NULL) {
#else
  if (std::freopen("/dev/null", "r", stdin) == NULL) {
#endif
    llvm::errs() << "Failed to close stdin" << '\n';
    exit(EXIT_FAILURE);
  }
  if (std::freopen(Stdout.c_str(), "a", stdout) == NULL) {
    llvm::errs() << "Failed to redirect stdout to " << Stdout << '\n';
    exit(EXIT_FAILURE);
  }
  if (std::freopen(Stderr.c_str(), "a", stderr) == NULL) {
    llvm::errs() << "Failed to redirect stderr to " << Stderr << '\n';
    exit(EXIT_FAILURE);
  }
}

// Creates unix socket for IPC with frontends
void ModuleBuildDaemonServer::createDaemonSocket() {
  while (true) {
    llvm::Expected<llvm::ListeningSocket> MaybeServerListener =
        llvm::ListeningSocket::createUnix(SocketPath);

    if (llvm::Error Err = MaybeServerListener.takeError()) {
      llvm::handleAllErrors(std::move(Err), [&](const llvm::StringError &SE) {
        std::error_code EC = SE.convertToErrorCode();

        // Exit successfully if the socket address is already in use. When
        // TUs are compiled in parallel, until the socket file is created, all
        // clang invocations will try to spawn a module build daemon.
#ifdef _WIN32
        if (EC.value() == WSAEADDRINUSE) {
#else
        if (EC == std::errc::address_in_use) {
#endif
          exit(EXIT_SUCCESS);
        } else if (EC == std::errc::file_exists) {
          if (std::remove(SocketPath.c_str()) != 0) {
            llvm::errs() << "Failed to remove " << SocketPath << ": "
                         << strerror(errno) << '\n';
            exit(EXIT_FAILURE);
          }
          // If a previous module build daemon invocation crashes, the socket
          // file will need to be removed before the address can be bound to
          logVerbose("Removing ineligible file: " + SocketPath);
        } else {
          llvm::errs() << "MBD failed to create unix socket: "
                       << SE.getMessage() << ": " << EC.message() << '\n';
          exit(EXIT_FAILURE);
        }
      });
    } else {
      logVerbose("MBD created and bound to socket at: " + SocketPath);
      ServerListener.emplace(std::move(*MaybeServerListener));
      break;
    }
  }
}

// Function submitted to thread pool with each frontend connection. Not
// responsible for closing frontend socket connections
void ModuleBuildDaemonServer::handleConnection(
    std::shared_ptr<llvm::raw_socket_stream> MovableConnection) {
  llvm::raw_socket_stream &Connection = *MovableConnection;

  // Read request from frontend
  llvm::Expected<HandshakeMsg> MaybeHandshakeMsg =
      readMsgStructFromSocket<HandshakeMsg>(Connection);
  if (!MaybeHandshakeMsg) {
    llvm::errs() << "MBD failed to read frontend request: "
                 << llvm::toString(MaybeHandshakeMsg.takeError()) << '\n';
    return;
  }

  // Send response to frontend
  HandshakeMsg Msg(ActionType::HANDSHAKE, StatusType::SUCCESS);
  if (llvm::Error WriteErr = writeMsgStructToSocket(Connection, Msg))
    llvm::errs() << "MBD failed to respond to frontend request: "
                 << llvm::toString(std::move(WriteErr)) << '\n';
  return;
}

void ModuleBuildDaemonServer::listenForClients() {
  llvm::DefaultThreadPool Pool;
  std::chrono::seconds DaemonTimeout(15);
  modifySignals(handleSignal);

  while (RunServiceLoop) {
    llvm::Expected<std::unique_ptr<llvm::raw_socket_stream>> MaybeConnection =
        ServerListener.value().accept(DaemonTimeout);

    if (llvm::Error Err = MaybeConnection.takeError()) {
      llvm::handleAllErrors(std::move(Err), [&](const llvm::StringError &SE) {
        std::error_code EC = SE.convertToErrorCode();

        if (EC == std::errc::timed_out) {
          RunServiceLoop = false;
          logVerbose("ListeningServer::accept timed out, shutting down");
        } else if (EC == std::errc::operation_canceled &&
                   RunServiceLoop == false) {
          logVerbose("Signal received, shutting down");
        } else
          llvm::errs() << "MBD failed to accept incoming connection: "
                       << SE.getMessage() << ": " << EC.message() << '\n';
      });
      continue;
    }

    // Connection must be copy constructable to be passed to Pool.async
    std::shared_ptr<llvm::raw_socket_stream> Connection(
        std::move(*MaybeConnection));
    Pool.async(handleConnection, Connection);
  }
}

// Module build daemon is spawned with the following command line:
//
// clang -cc1modbuildd [<path>] [-v]
//
// OPTIONS
//   <path>
//       Specifies the path to all the files created by the module build daemon.
//       If provided, <path> should immediately follow -cc1modbuildd.
//
//   -v
//       Provides verbose debug information.
//
// NOTES
//     The arguments <path> and -v are optional. If <path> is not provided then
//     BasePath will be /tmp/clang-<BLAKE3HashOfClangFullVersion>
//
int cc1modbuildd_main(llvm::ArrayRef<const char *> Argv) {
  // -cc1modbuildd is sliced away when Argv is pased to cc1modbuildd_main
  if (find(Argv, llvm::StringRef("-v")) != Argv.end())
    LogVerbose = true;

  std::string BasePath;
  // If an argument exists and it is not -v then it must be a BasePath
  if (!Argv.empty() && strcmp(Argv[0], "-v") != 0)
    BasePath = Argv[0];
  else
    BasePath = getBasePath();

  if (!validBasePathLength(BasePath)) {
    llvm::errs() << "BasePath '" << BasePath
                 << "' is longer then the max length of "
                 << std::to_string(BasePathMaxLength) << '\n';
    return EXIT_FAILURE;
  }

  llvm::sys::fs::create_directories(BasePath);

  {
    ModuleBuildDaemonServer Daemon(BasePath);

    // Used to handle signals
    DaemonPtr = &Daemon;

    Daemon.setupDaemonEnv();
    Daemon.createDaemonSocket();
    Daemon.listenForClients();

    // Prevents the signal handler from being called after the
    // ModuleBuildDaemonServer is destructed. The daemon is shutting down and
    // the program is about to return so signals can be ignored
    modifySignals(SIG_IGN);
  }

  return EXIT_SUCCESS;
}
