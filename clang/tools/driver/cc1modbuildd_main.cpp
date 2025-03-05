//===------- cc1modbuildd_main.cpp - Clang CC1 Module Build Daemon --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/DiagnosticCategories.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningService.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningTool.h"
#include "clang/Tooling/DependencyScanning/ModuleDepCollector.h"
#include "clang/Tooling/ModuleBuildDaemon/SocketSupport.h"
#include "clang/Tooling/ModuleBuildDaemon/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_socket_stream.h"

#include <csignal>
#include <errno.h>
#include <fstream>
#include <mutex>
#include <optional>
#include <signal.h>
#include <sstream>
#include <stdbool.h>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <type_traits>
#include <unistd.h>
#include <unordered_map>

using namespace llvm;
using namespace clang::tooling::dependencies;
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

enum class BuildStatus { WAITING, BUILDING, BUILT };

struct ModuleIDHash {
  std::size_t
  operator()(const clang::tooling::dependencies::ModuleID &ID) const {
    return llvm::hash_value(ID);
  }
};

struct ModuleBuildInfo {
  const ModuleDeps Info;
  BuildStatus ModuleBuildStatus;
};

// Thread safe hash map that stores dependency and build information
class DependencyBuildData {
public:
  void insert(ModuleID Key, ModuleBuildInfo Value) {
    std::lock_guard<std::mutex> lock(Mutex);
    HashMap.insert({Key, Value});
  }

  std::optional<std::reference_wrapper<ModuleBuildInfo>> get(ModuleID Key) {
    std::lock_guard<std::mutex> lock(Mutex);
    if (auto search = HashMap.find(Key); search != HashMap.end())
      return std::ref(search->second);
    return std::nullopt;
  }

  bool updateBuildStatus(ModuleID Key, BuildStatus newStatus) {
    std::lock_guard<std::mutex> lock(Mutex);
    if (auto search = HashMap.find(Key); search != HashMap.end()) {
      search->second.ModuleBuildStatus = newStatus;
      return true;
    }
    return false;
  }

  void print() {
    unbuff_outs() << "printing hash table keys" << '\n';
    for (const auto &i : HashMap) {
      unbuff_outs() << "Module: " << i.first.ModuleName << '\n';
      unbuff_outs() << "Dependencies: ";
      for (const auto &Dep : i.second.Info.ClangModuleDeps)
        unbuff_outs() << Dep.ModuleName << ", ";
      unbuff_outs() << '\n';
    }
  }

private:
  std::unordered_map<ModuleID, ModuleBuildInfo, ModuleIDHash> HashMap;
  std::mutex Mutex;
};

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

  ~ModuleBuildDaemonServer() { shutdownDaemon(); }

  void setupDaemonEnv();
  void createDaemonSocket();
  void listenForClients();

  static void handleRegister(llvm::raw_socket_stream &Client,
                             RegisterMsg ClientRequest);
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

static DependencyBuildData DaemonBuildData;

static Expected<TranslationUnitDeps> scanTranslationUnit(RegisterMsg Request) {

  DependencyScanningService Service(ScanningMode::DependencyDirectivesScan,
                                    ScanningOutputFormat::Full);

  DependencyScanningTool Tool(Service);
  llvm::DenseSet<ModuleID> AlreadySeenModules;
  auto LookupOutput = [&](const ModuleID &MID, ModuleOutputKind MOK) {
    return "/tmp/" + MID.ContextHash;
  };

  // Add executable path to cc1 command line for dependency scanner
  std::vector<std::string> ScannerCommandLine;
  ScannerCommandLine.push_back(Request.ExecutablePath.value());
  ScannerCommandLine.insert(ScannerCommandLine.end(),
                            Request.CC1CommandLine.value().begin(),
                            Request.CC1CommandLine.value().end());

  Expected<TranslationUnitDeps> MaybeTUDeps =
      Tool.getTranslationUnitDependencies(ScannerCommandLine,
                                          Request.WorkingDirectory.value(),
                                          AlreadySeenModules, LookupOutput);

  if (!MaybeTUDeps)
    return std::move(MaybeTUDeps.takeError());

  return std::move(*MaybeTUDeps);
}

static void storeScanResults(const TranslationUnitDeps &Results) {

  if (Results.ModuleGraph.empty())
    return;

  // Insert children
  for (const ModuleDeps &MD : Results.ModuleGraph)
    DaemonBuildData.insert(MD.ID, {MD, BuildStatus::WAITING});
}

// Remove -fmodule-build-daemon and add -fno-implicit-modules to command line.
// Return value can either be a std::vector of std::string or StringRef
template <typename T>
static std::vector<T> modifyCC1(const std::vector<std::string> &CommandLine) {
  static_assert(std::is_same_v<T, std::string> || std::is_same_v<T, StringRef>);

  std::vector<T> ReturnArgs;
  ReturnArgs.reserve(CommandLine.size());

  for (const auto &Arg : CommandLine) {
    if (Arg != "-fmodule-build-daemon")
      ReturnArgs.emplace_back(Arg);
  }
  ReturnArgs.emplace_back("-fno-implicit-modules");
  return ReturnArgs;
}

// TODO: Return llvm::Error
static void precompileModuleID(const StringRef Executable, const ModuleID ID) {
  unbuff_outs() << "module " << ID.ModuleName << " will be built" << '\n';

  std::optional<std::reference_wrapper<ModuleBuildInfo>> MaybeDeps =
      DaemonBuildData.get(ID);
  if (!MaybeDeps)
    return;
  ModuleBuildInfo &Deps = MaybeDeps->get();

  // TODO: look into making getBuildArguments a const method
  ModuleDeps &NonConstDepsInfo = const_cast<ModuleDeps &>(Deps.Info);
  const std::vector<std::string> &Args = NonConstDepsInfo.getBuildArguments();
  std::vector<std::string> NonConstArgs =
      const_cast<std::vector<std::string> &>(Args);

  unbuff_outs() << "original command line" << '\n';
  for (const auto &Arg : Args)
    unbuff_outs() << Arg << " ";
  unbuff_outs() << "\n\n";

  const std::vector<StringRef> ProcessedArgs =
      modifyCC1<StringRef>(NonConstArgs);

  std::vector<StringRef> ExecuteCommandLine;
  ExecuteCommandLine.push_back(Executable);
  ExecuteCommandLine.insert(ExecuteCommandLine.end(), ProcessedArgs.begin(),
                            ProcessedArgs.end());

  unbuff_outs() << "new command line" << '\n';
  for (const auto &Arg : NonConstArgs)
    unbuff_outs() << Arg << " ";
  unbuff_outs() << "\n";

  // TODO: Handle error code returned from ExecuteAndWait
  llvm::sys::ExecuteAndWait(Executable,
                            ArrayRef<StringRef>(ExecuteCommandLine));
  DaemonBuildData.updateBuildStatus(ID, BuildStatus::BUILT);

  unbuff_outs() << "module " << ID.ModuleName << " finished building" << '\n';
  unbuff_outs() << "\n\n";

  return;
}

// TODO: implement concurrent approach
// can only handle one translation unit at a time
static void buildModuleID(const StringRef Executable, const ModuleID ID) {

  std::optional<std::reference_wrapper<ModuleBuildInfo>> MaybeDeps =
      DaemonBuildData.get(ID);
  if (!MaybeDeps)
    return;
  ModuleBuildInfo &Deps = MaybeDeps->get();

  if (Deps.ModuleBuildStatus == BuildStatus::BUILT)
    return;

  for (const ModuleID &Dep : Deps.Info.ClangModuleDeps)
    buildModuleID(Executable, Dep);

  // Do not build the root ID aka the registered translation unit
  if (ID.ModuleName.empty())
    return;

  precompileModuleID(Executable, ID);
  return;
}

// Takes a client request in the form of a cc1modbuildd::SocketMsg and
// returns an updated cc1 command line for the registered cc1 invocation
// after building all modular dependencies
static Expected<std::vector<std::string>>
processRegisterRequest(RegisterMsg Request) {

  Expected<TranslationUnitDeps> MaybeTUDeps = scanTranslationUnit(Request);
  if (!MaybeTUDeps)
    return std::move(MaybeTUDeps.takeError());
  const TranslationUnitDeps TUDeps = std::move(*MaybeTUDeps);

  // For now write dependencies to log file
  for (const auto &Dep : TUDeps.FileDeps)
    unbuff_outs() << Dep << '\n';

  // If TU does not depend on modules then return command line as is
  if (TUDeps.ModuleGraph.empty())
    return Request.CC1CommandLine.value();

  unbuff_outs() << "modules detected" << '\n';
  storeScanResults(TUDeps);
  DaemonBuildData.print();

  // Build all direct and transitive dependencies by iterating over direct
  // dependencies
  for (const ModuleID &Dep : TUDeps.ClangModuleDeps)
    buildModuleID(Request.ExecutablePath.value(), Dep);

  return modifyCC1<std::string>(TUDeps.Commands[0].Arguments);
}

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

void ModuleBuildDaemonServer::handleRegister(llvm::raw_socket_stream &Client,
                                             RegisterMsg ClientRequest) {

  Expected<std::vector<std::string>> MaybeExplicitCC1 =
      processRegisterRequest(ClientRequest);

  // if getUpdatedCC1 fails emit error
  if (!MaybeExplicitCC1) {

    RegisterMsg Msg(ActionType::REGISTER, StatusType::FAILURE);
    llvm::Error RegisterFailureWriteErr = writeMsgStructToSocket(Client, Msg);

    if (RegisterFailureWriteErr) {
      writeError(llvm::joinErrors(std::move(RegisterFailureWriteErr),
                                  std::move(MaybeExplicitCC1.takeError())),
                 "Failed to process register request and was unable to notify "
                 "clang infocation: ");
      return;
    }
    writeError(MaybeExplicitCC1.takeError(),
               "Failed to process register request: ");
    return;
  }

  // getUpdateCC1 success
  std::vector<std::string> ExplicitCC1 = std::move(*MaybeExplicitCC1);

  unbuff_outs() << "modified command line for translation unit" << '\n';
  for (const auto &Arg : ExplicitCC1)
    unbuff_outs() << Arg << " ";
  unbuff_outs() << "\n";

  // Send new CC1 command line to waiting clang invocation
  RegisterMsg Msg(ActionType::REGISTER, StatusType::SUCCESS,
                  ClientRequest.WorkingDirectory, ClientRequest.ExecutablePath,
                  ExplicitCC1);

  llvm::Error RegisterSuccessWriteErr = writeMsgStructToSocket(Client, Msg);

  if (RegisterSuccessWriteErr) {
    writeError(std::move(RegisterSuccessWriteErr),
               "Failed to notify clang invocation that register request was a "
               "success: ");
    return;
  }
  return;
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
  if (llvm::Error WriteErr = writeMsgStructToSocket(Connection, Msg)) {
    llvm::errs() << "MBD failed to respond to frontend request: "
                 << llvm::toString(std::move(WriteErr)) << '\n';
    return;
  }

  // Read request from frontend
  Expected<RegisterMsg> MaybeRegister =
      readMsgStructFromSocket<RegisterMsg>(Connection);
  if (!MaybeRegister) {
    llvm::errs() << "Failed to read registration message from socket: "
                 << llvm::toString(std::move(MaybeRegister.takeError()))
                 << '\n';
    return;
  }

  RegisterMsg Register = std::move(*MaybeRegister);
  handleRegister(Connection, Register);

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
// clang -cc1modbuildd <path> -v
//
// <path> defines the location of all files created by the module build daemon
// and should follow the format /path/to/dir. For example, `clang
// -cc1modbuildd /tmp/` creates a socket file at `/tmp/mbd.sock`. /tmp is also
// valid.
//
// When module build daemons are spawned by cc1 invocations, <path> follows
// the format /tmp/clang-<BLAKE3HashOfClangFullVersion>
//
// -v is optional and provides debug information
//
int cc1modbuildd_main(ArrayRef<const char *> Argv) {
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
