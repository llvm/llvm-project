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
#include <fstream>
#include <optional>
#include <string>
#include <system_error>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

using namespace llvm;
using namespace clang::tooling::cc1modbuildd;

// Create unbuffered STDOUT stream so that any logging done by the module build
// daemon can be viewed without having to terminate the process
static raw_fd_ostream &unbuff_outs() {
  static raw_fd_ostream S(fileno(stdout), false, true);
  return S;
}

static bool VerboseLog = false;
static void verboseLog(const llvm::Twine &message) {
  if (VerboseLog) {
    unbuff_outs() << message << '\n';
  }
}

namespace {

class ModuleBuildDaemonServer {
public:
  SmallString<256> SocketPath;
  SmallString<256> STDERR;
  SmallString<256> STDOUT;

  ModuleBuildDaemonServer(StringRef Path)
      : SocketPath(Path), STDERR(Path), STDOUT(Path) {
    llvm::sys::path::append(SocketPath, SOCKET_FILE_NAME);
    llvm::sys::path::append(STDOUT, STDOUT_FILE_NAME);
    llvm::sys::path::append(STDERR, STDERR_FILE_NAME);
  }

  ~ModuleBuildDaemonServer() { shutdownDaemon(); }

  void setupDaemonEnv();
  void createDaemonSocket();
  void listenForClients();

  static void handleConnection(std::shared_ptr<raw_socket_stream> Connection);

  // TODO: modify so when shutdownDaemon is called the daemon stops accepting
  // new client connections and waits for all existing client connections to
  // terminate before closing the file descriptor and exiting
  void shutdownDaemon() {
    RunServiceLoop = false;
    if (ServerListener.has_value())
      ServerListener.value().shutdown();
  }

private:
  std::atomic<bool> RunServiceLoop = true;
  std::optional<llvm::ListeningSocket> ServerListener;
};

// Used to handle signals
ModuleBuildDaemonServer *DaemonPtr = nullptr;
void handleSignal(int) { DaemonPtr->shutdownDaemon(); }
} // namespace

// Sets up file descriptors and signals for module build daemon
void ModuleBuildDaemonServer::setupDaemonEnv() {

#ifdef _WIN32
  freopen("NUL", "r", stdin);
#else
  close(STDIN_FILENO);
#endif

  freopen(STDOUT.c_str(), "a", stdout);
  freopen(STDERR.c_str(), "a", stderr);

  if (std::signal(SIGTERM, handleSignal) == SIG_ERR) {
    errs() << "failed to handle SIGTERM" << '\n';
    exit(EXIT_FAILURE);
  }

  if (std::signal(SIGINT, handleSignal) == SIG_ERR) {
    errs() << "failed to handle SIGINT" << '\n';
    exit(EXIT_FAILURE);
  }

// TODO: Figure out how to do this on windows
#ifdef SIGHUP
  if (::signal(SIGHUP, SIG_IGN) == SIG_ERR) {
    errs() << "failed to handle SIGHUP" << '\n';
    exit(EXIT_FAILURE);
  }
#endif
}

// Creates unix socket for IPC with frontends
void ModuleBuildDaemonServer::createDaemonSocket() {

  Expected<ListeningSocket> MaybeServerListener =
      llvm::ListeningSocket::createUnix(SocketPath);

  if (llvm::Error Err = MaybeServerListener.takeError()) {
    llvm::handleAllErrors(std::move(Err), [&](const llvm::StringError &SE) {
      std::error_code EC = SE.convertToErrorCode();
      // Exit successfully if the socket address is already in use. When
      // translation units are compiled in parallel, until the socket file is
      // created, all clang invocations will try to spawn a module build daemon.
#ifdef _WIN32
      if (EC.value() == WSAEADDRINUSE) {
#else
      if (EC == std::errc::address_in_use) {
#endif
        exit(EXIT_SUCCESS);
      } else {
        llvm::errs() << "MBD failed to create unix socket: " << SE.message()
                     << EC.message() << '\n';
        exit(EXIT_FAILURE);
      }
    });
  }

  verboseLog("mbd created and binded to socket at: " + SocketPath);
  ServerListener.emplace(std::move(*MaybeServerListener));
}

// Function submitted to thread pool with each frontend connection. Not
// responsible for closing frontend socket connections
void ModuleBuildDaemonServer::handleConnection(
    std::shared_ptr<llvm::raw_socket_stream> MovableConnection) {

  llvm::raw_socket_stream &Connection = *MovableConnection;

  // Read request from frontend
  Expected<HandshakeMsg> MaybeHandshakeMsg =
      readMsgStructFromSocket<HandshakeMsg>(Connection);
  if (!MaybeHandshakeMsg) {
    errs() << "MBD failed to read frontend request: "
           << llvm::toString(MaybeHandshakeMsg.takeError()) << '\n';
    return;
  }

  // Send response to frontend
  HandshakeMsg Msg(ActionType::HANDSHAKE, StatusType::SUCCESS);
  if (llvm::Error WriteErr = writeMsgStructToSocket(Connection, Msg)) {
    errs() << "MBD failed to respond to frontend request: "
           << llvm::toString(std::move(WriteErr)) << '\n';
    return;
  }
  return;
}

void ModuleBuildDaemonServer::listenForClients() {

  llvm::ThreadPool Pool;
  std::chrono::microseconds DaemonTimeout(15 * MICROSEC_IN_SEC);

  while (RunServiceLoop) {
    Expected<std::unique_ptr<raw_socket_stream>> MaybeConnection =
        ServerListener.value().accept(DaemonTimeout);

    if (llvm::Error Err = MaybeConnection.takeError()) {

      llvm::handleAllErrors(std::move(Err), [&](const llvm::StringError &SE) {
        std::error_code EC = SE.convertToErrorCode();

        if (EC == std::errc::timed_out) {
          RunServiceLoop = false;
          verboseLog("ListeningServer::accept timed out, shutting down");
        } else if (EC == std::errc::interrupted && RunServiceLoop == false) {
          verboseLog("Signal received, shutting down");
        } else
          errs() << "MBD failed to accept incoming connection: "
                 << SE.getMessage() << ": " << EC.message() << '\n';
      });

      continue;
    }

    // Connection must be copy constructable to be passed to Pool.async
    std::shared_ptr<raw_socket_stream> Connection(std::move(*MaybeConnection));
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
int cc1modbuildd_main(ArrayRef<const char *> Argv) {

  // -cc1modbuildd is sliced away when Argv is pased to cc1modbuildd_main
  if (find(Argv, StringRef("-v")) != Argv.end())
    VerboseLog = true;

  std::string BasePath;
  // If an argument exists and it is not -v then it must be a BasePath
  if (!Argv.empty() && strcmp(Argv[0], "-v") != 0)
    BasePath = Argv[0];
  else
    BasePath = getBasePath();

  if (!validBasePathLength(BasePath)) {
    errs() << "BasePath '" << BasePath << "' is longer then the max length of "
           << std::to_string(BASEPATH_MAX_LENGTH) << '\n';
    return 1;
  }

  llvm::sys::fs::create_directories(BasePath);
  ModuleBuildDaemonServer Daemon(BasePath);

  // Used to handle signals
  DaemonPtr = &Daemon;

  Daemon.setupDaemonEnv();
  Daemon.createDaemonSocket();
  Daemon.listenForClients();

  return EXIT_SUCCESS;
}
