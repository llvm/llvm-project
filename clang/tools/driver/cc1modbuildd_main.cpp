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
#include "clang/Tooling/ModuleBuildDaemon/SocketMsgSupport.h"
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

// TODO: Make portable
#if LLVM_ON_UNIX

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
using namespace clang;
using namespace tooling::dependencies;
using namespace cc1modbuildd;

// Create unbuffered STDOUT stream so that any logging done by module build
// daemon can be viewed without having to terminate the process
static raw_fd_ostream &unbuff_outs() {
  static raw_fd_ostream S(STDOUT_FILENO, false, true);
  return S;
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
  SmallString<128> BasePath;
  SmallString<128> SocketPath;
  SmallString<128> PidPath;

  ModuleBuildDaemonServer(SmallString<128> Path, ArrayRef<const char *> Argv)
      : BasePath(Path), SocketPath(Path) {
    llvm::sys::path::append(SocketPath, SOCKET_FILE_NAME);
  }

  ~ModuleBuildDaemonServer() { shutdownDaemon(SIGTERM); }

  int forkDaemon();
  int launchDaemon();
  int listenForClients();

  static void handleClient(int Client);
  static void handleRegister(int Client, RegisterMsg ClientRequest);

  void shutdownDaemon(int signal) {
    unlink(SocketPath.c_str());
    shutdown(ListenSocketFD, SHUT_RD);
    close(ListenSocketFD);
    exit(EXIT_SUCCESS);
  }

private:
  // Initializes and returns DiagnosticsEngine
  pid_t Pid = -1;
  int ListenSocketFD = -1;
};

// Required to handle SIGTERM by calling Shutdown
ModuleBuildDaemonServer *DaemonPtr = nullptr;
void handleSignal(int Signal) {
  if (DaemonPtr != nullptr) {
    DaemonPtr->shutdownDaemon(Signal);
  }
}
} // namespace

static bool verbose = false;
static void verbose_print(const llvm::Twine &message) {
  if (verbose) {
    unbuff_outs() << message << '\n';
  }
}

static DependencyBuildData DaemonBuildData;

static Expected<TranslationUnitDeps>
scanTranslationUnit(cc1modbuildd::RegisterMsg Request) {

  DependencyScanningService Service(ScanningMode::DependencyDirectivesScan,
                                    ScanningOutputFormat::Full,
                                    /*OptimizeArgs*/ false,
                                    /*EagerLoadModules*/ false);

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
processRegisterRequest(cc1modbuildd::RegisterMsg Request) {

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

// Forks and detaches process, creating module build daemon
int ModuleBuildDaemonServer::forkDaemon() {

  pid_t pid = fork();

  if (pid < 0) {
    exit(EXIT_FAILURE);
  }
  if (pid > 0) {
    exit(EXIT_SUCCESS);
  }

  Pid = getpid();

  // close(STDIN_FILENO);
  // close(STDOUT_FILENO);
  // close(STDERR_FILENO);

  // SmallString<128> STDOUT = BasePath;
  // llvm::sys::path::append(STDOUT, STDOUT_FILE_NAME);
  // freopen(STDOUT.c_str(), "a", stdout);

  // SmallString<128> STDERR = BasePath;
  // llvm::sys::path::append(STDERR, STDERR_FILE_NAME);
  // freopen(STDERR.c_str(), "a", stderr);

  if (signal(SIGTERM, handleSignal) == SIG_ERR) {
    errs() << "failed to handle SIGTERM" << '\n';
    exit(EXIT_FAILURE);
  }
  if (signal(SIGHUP, SIG_IGN) == SIG_ERR) {
    errs() << "failed to ignore SIGHUP" << '\n';
    exit(EXIT_FAILURE);
  }
  if (setsid() == -1) {
    errs() << "setsid failed" << '\n';
    exit(EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}

// Creates unix socket for IPC with module build daemon
int ModuleBuildDaemonServer::launchDaemon() {

  // new socket
  if ((ListenSocketFD = socket(AF_UNIX, SOCK_STREAM, 0)) == -1) {
    std::perror("Socket create error: ");
    exit(EXIT_FAILURE);
  }

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(struct sockaddr_un));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, SocketPath.c_str(), sizeof(addr.sun_path) - 1);

  // bind to local address
  if (bind(ListenSocketFD, (struct sockaddr *)&addr, sizeof(addr)) == -1) {

    // If the socket address is already in use, exit because another module
    // build daemon has successfully launched. When translation units are
    // compiled in parallel, until the socket file is created, all clang
    // invocations will spawn a module build daemon.
    if (errno == EADDRINUSE) {
      close(ListenSocketFD);
      exit(EXIT_SUCCESS);
    }
    std::perror("Socket bind error: ");
    exit(EXIT_FAILURE);
  }
  verbose_print("mbd created and binded to socket address at: " + SocketPath);

  // set socket to accept incoming connection request
  unsigned MaxBacklog = llvm::hardware_concurrency().compute_thread_count();
  if (listen(ListenSocketFD, MaxBacklog) == -1) {
    std::perror("Socket listen error: ");
    exit(EXIT_FAILURE);
  }

  return 0;
}

void ModuleBuildDaemonServer::handleRegister(int Client,
                                             RegisterMsg ClientRequest) {

  Expected<std::vector<std::string>> MaybeExplicitCC1 =
      processRegisterRequest(ClientRequest);

  // if getUpdatedCC1 fails emit error
  if (!MaybeExplicitCC1) {

    RegisterMsg Msg(ActionType::REGISTER, StatusType::FAILURE);
    llvm::Error RegisterFailureWriteErr = writeSocketMsgToSocket(Msg, Client);

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

  llvm::Error RegisterSuccessWriteErr = writeSocketMsgToSocket(Msg, Client);

  if (RegisterSuccessWriteErr) {
    writeError(std::move(RegisterSuccessWriteErr),
               "Failed to notify clang invocation that register request was a "
               "success: ");
    return;
  }
  return;
}

// Function submitted to thread pool with each client connection. Not
// responsible for closing client connections
void ModuleBuildDaemonServer::handleClient(int Client) {

  // Read handshake from client
  Expected<HandshakeMsg> MaybeHandshake =
      readSocketMsgFromSocket<HandshakeMsg>(Client);

  if (!MaybeHandshake) {
    writeError(MaybeHandshake.takeError(),
               "Failed to read handshake message from socket: ");
    return;
  }

  // Handle HANDSHAKE
  RegisterMsg Msg(ActionType::HANDSHAKE, StatusType::SUCCESS);
  llvm::Error WriteErr = writeSocketMsgToSocket(Msg, Client);

  if (WriteErr) {
    writeError(std::move(WriteErr),
               "Failed to notify client that handshake was received");
    return;
  }

  // Read register request from client
  Expected<RegisterMsg> MaybeRegister =
      readSocketMsgFromSocket<RegisterMsg>(Client);

  if (!MaybeRegister) {
    writeError(MaybeRegister.takeError(),
               "Failed to read registration message from socket: ");
    return;
  }

  RegisterMsg Register = std::move(*MaybeRegister);
  handleRegister(Client, Register);
  return;
}

int ModuleBuildDaemonServer::listenForClients() {

  llvm::ThreadPool Pool;
  int Client;

  while (true) {

    if ((Client = accept(ListenSocketFD, NULL, NULL)) == -1) {
      std::perror("Socket accept error: ");
      continue;
    }

    Pool.async(handleClient, Client);
  }
  return 0;
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

  if (Argv.size() < 1) {
    outs() << "spawning a module build daemon requies a command line format of "
              "`clang -cc1modbuildd <path>`. <path> defines where the module "
              "build daemon will create files"
           << '\n';
    return 1;
  }

  // Where to store log files and socket address
  // TODO: Add check to confirm BasePath is directory
  SmallString<128> BasePath(Argv[0]);
  llvm::sys::fs::create_directories(BasePath);
  ModuleBuildDaemonServer Daemon(BasePath, Argv);

  // Used to handle signals
  DaemonPtr = &Daemon;

  if (find(Argv, StringRef("-v")) != Argv.end())
    verbose = true;

  Daemon.forkDaemon();
  Daemon.launchDaemon();
  Daemon.listenForClients();

  return 0;
}

#endif // LLVM_ON_UNIX
