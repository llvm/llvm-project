//===--- tools/clang-repl/ClangRepl.cpp - clang-repl - the Clang REPL -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements a REPL tool on top of clang.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Interpreter/CodeCompletion.h"
#include "clang/Interpreter/Interpreter.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/SimpleRemoteEPC.h"
#include "llvm/LineEditor/LineEditor.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h" // llvm_shutdown
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"
#include <optional>
#include <sys/types.h>

#ifdef LLVM_ON_UNIX
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#endif // LLVM_ON_UNIX

// Disable LSan for this test.
// FIXME: Re-enable once we can assume GCC 13.2 or higher.
// https://llvm.org/github.com/llvm/llvm-project/issues/67586.
#if LLVM_ADDRESS_SANITIZER_BUILD || LLVM_HWADDRESS_SANITIZER_BUILD
#include <sanitizer/lsan_interface.h>
LLVM_ATTRIBUTE_USED int __lsan_is_turned_off() { return 1; }
#endif

static llvm::cl::opt<bool> CudaEnabled("cuda", llvm::cl::Hidden);
static llvm::cl::opt<std::string> CudaPath("cuda-path", llvm::cl::Hidden);
static llvm::cl::opt<std::string> OffloadArch("offload-arch", llvm::cl::Hidden);

static llvm::cl::list<std::string>
    ClangArgs("Xcc",
              llvm::cl::desc("Argument to pass to the CompilerInvocation"),
              llvm::cl::CommaSeparated);
static llvm::cl::opt<bool> OptHostSupportsJit("host-supports-jit",
                                              llvm::cl::Hidden);
static llvm::cl::list<std::string> OptInputs(llvm::cl::Positional,
                                             llvm::cl::desc("[code to run]"));

static llvm::cl::OptionCategory OOPCategory(
    "Out-of-process Execution Options (Only available on ELF/Linux)");
static llvm::cl::opt<std::string> OutOfProcessExecutor(
    "oop-executor",
    llvm::cl::desc("Launch an out-of-process executor to run code"),
    llvm::cl::ValueOptional, llvm::cl::cat(OOPCategory));
static llvm::cl::opt<std::string> OutOfProcessExecutorConnect(
    "oop-executor-connect",
    llvm::cl::desc("Connect to an out-of-process executor via TCP"),
    llvm::cl::cat(OOPCategory));
static llvm::cl::opt<std::string>
    OrcRuntimePath("orc-runtime", llvm::cl::desc("Path to the ORC runtime"),
                   llvm::cl::cat(OOPCategory));

static void LLVMErrorHandler(void *UserData, const char *Message,
                             bool GenCrashDiag) {
  auto &Diags = *static_cast<clang::DiagnosticsEngine *>(UserData);

  Diags.Report(clang::diag::err_fe_error_backend) << Message;

  // Run the interrupt handlers to make sure any special cleanups get done, in
  // particular that we remove files registered with RemoveFileOnSignal.
  llvm::sys::RunInterruptHandlers();

  // We cannot recover from llvm errors.  When reporting a fatal error, exit
  // with status 70 to generate crash diagnostics.  For BSD systems this is
  // defined as an internal software error. Otherwise, exit with status 1.

  exit(GenCrashDiag ? 70 : 1);
}

// If we are running with -verify a reported has to be returned as unsuccess.
// This is relevant especially for the test suite.
static int checkDiagErrors(const clang::CompilerInstance *CI, bool HasError) {
  unsigned Errs = CI->getDiagnostics().getClient()->getNumErrors();
  if (CI->getDiagnosticOpts().VerifyDiagnostics) {
    // If there was an error that came from the verifier we must return 1 as
    // an exit code for the process. This will make the test fail as expected.
    clang::DiagnosticConsumer *Client = CI->getDiagnostics().getClient();
    Client->EndSourceFile();
    Errs = Client->getNumErrors();

    // The interpreter expects BeginSourceFile/EndSourceFiles to be balanced.
    Client->BeginSourceFile(CI->getLangOpts(), &CI->getPreprocessor());
  }
  return (Errs || HasError) ? EXIT_FAILURE : EXIT_SUCCESS;
}

struct ReplListCompleter {
  clang::IncrementalCompilerBuilder &CB;
  clang::Interpreter &MainInterp;
  ReplListCompleter(clang::IncrementalCompilerBuilder &CB,
                    clang::Interpreter &Interp)
      : CB(CB), MainInterp(Interp){};

  std::vector<llvm::LineEditor::Completion> operator()(llvm::StringRef Buffer,
                                                       size_t Pos) const;
  std::vector<llvm::LineEditor::Completion>
  operator()(llvm::StringRef Buffer, size_t Pos, llvm::Error &ErrRes) const;
};

std::vector<llvm::LineEditor::Completion>
ReplListCompleter::operator()(llvm::StringRef Buffer, size_t Pos) const {
  auto Err = llvm::Error::success();
  auto res = (*this)(Buffer, Pos, Err);
  if (Err)
    llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(), "error: ");
  return res;
}

std::vector<llvm::LineEditor::Completion>
ReplListCompleter::operator()(llvm::StringRef Buffer, size_t Pos,
                              llvm::Error &ErrRes) const {
  std::vector<llvm::LineEditor::Completion> Comps;
  std::vector<std::string> Results;

  auto CI = CB.CreateCpp();
  if (auto Err = CI.takeError()) {
    ErrRes = std::move(Err);
    return {};
  }

  size_t Lines =
      std::count(Buffer.begin(), std::next(Buffer.begin(), Pos), '\n') + 1;
  auto Interp = clang::Interpreter::create(std::move(*CI));

  if (auto Err = Interp.takeError()) {
    // log the error and returns an empty vector;
    ErrRes = std::move(Err);

    return {};
  }

  codeComplete(
      const_cast<clang::CompilerInstance *>((*Interp)->getCompilerInstance()),
      Buffer, Lines, Pos + 1, MainInterp.getCompilerInstance(), Results);

  size_t space_pos = Buffer.rfind(" ");
  llvm::StringRef Prefix;
  if (space_pos == llvm::StringRef::npos) {
    Prefix = Buffer;
  } else {
    Prefix = Buffer.substr(space_pos + 1);
  }

  for (auto c : Results) {
    if (c.find(Prefix) == 0)
      Comps.push_back(llvm::LineEditor::Completion(c.substr(Prefix.size()), c));
  }
  return Comps;
}

static llvm::Error sanitizeOopArguments(const char *ArgV0) {
  llvm::Triple SystemTriple(llvm::sys::getProcessTriple());
  if ((OutOfProcessExecutor.getNumOccurrences() ||
       OutOfProcessExecutorConnect.getNumOccurrences()) &&
      (!SystemTriple.isOSBinFormatELF()))
    return llvm::make_error<llvm::StringError>(
        "Out-process-executors are currently only supported on ELF",
        llvm::inconvertibleErrorCode());

  // Only one of -oop-executor and -oop-executor-connect can be used.
  if (!!OutOfProcessExecutor.getNumOccurrences() &&
      !!OutOfProcessExecutorConnect.getNumOccurrences())
    return llvm::make_error<llvm::StringError>(
        "Only one of -" + OutOfProcessExecutor.ArgStr + " and -" +
            OutOfProcessExecutorConnect.ArgStr + " can be specified",
        llvm::inconvertibleErrorCode());

  // If -oop-executor was used but no value was specified then use a sensible
  // default.
  if (!!OutOfProcessExecutor.getNumOccurrences() &&
      OutOfProcessExecutor.empty()) {
    llvm::SmallString<256> OOPExecutorPath(llvm::sys::fs::getMainExecutable(
        ArgV0, reinterpret_cast<void *>(&sanitizeOopArguments)));
    llvm::sys::path::remove_filename(OOPExecutorPath);
    llvm::sys::path::append(OOPExecutorPath, "llvm-jitlink-executor");
    OutOfProcessExecutor = OOPExecutorPath.str().str();
  }

  // Out-of-process executors must run with the ORC runtime for destructor
  // support.
  if (OrcRuntimePath.empty() &&
      (OutOfProcessExecutor.getNumOccurrences() ||
       OutOfProcessExecutorConnect.getNumOccurrences())) {
    llvm::SmallString<256> OrcPath(llvm::sys::fs::getMainExecutable(
        ArgV0, reinterpret_cast<void *>(&sanitizeOopArguments)));
    llvm::sys::path::remove_filename(OrcPath); // Remove clang-repl filename.
    llvm::sys::path::remove_filename(OrcPath); // Remove ./bin directory.
    llvm::sys::path::append(
        OrcPath, "lib/clang/18/lib/x86_64-unknown-linux-gnu/liborc_rt.a");
    OrcRuntimePath = OrcPath.str().str();
  }

  return llvm::Error::success();
}

static llvm::Expected<std::unique_ptr<llvm::orc::ExecutorProcessControl>>
launchExecutor() {
#ifndef LLVM_ON_UNIX
  // FIXME: Add support for Windows.
  return make_error<StringError>("-" + OutOfProcessExecutor.ArgStr +
                                     " not supported on non-unix platforms",
                                 inconvertibleErrorCode());
#elif !LLVM_ENABLE_THREADS
  return make_error<StringError>(
      "-" + OutOfProcessExecutor.ArgStr +
          " requires threads, but LLVM was built with "
          "LLVM_ENABLE_THREADS=Off",
      inconvertibleErrorCode());
#else
  constexpr int ReadEnd = 0;
  constexpr int WriteEnd = 1;

  // Pipe FDs.
  int ToExecutor[2];
  int FromExecutor[2];

  pid_t ChildPID;

  // Create pipes to/from the executor..
  if (pipe(ToExecutor) != 0 || pipe(FromExecutor) != 0)
    return llvm::make_error<llvm::StringError>(
        "Unable to create pipe for executor", llvm::inconvertibleErrorCode());

  ChildPID = fork();

  if (ChildPID == 0) {
    // In the child...

    // Close the parent ends of the pipes
    close(ToExecutor[WriteEnd]);
    close(FromExecutor[ReadEnd]);

    // Execute the child process.
    std::unique_ptr<char[]> ExecutorPath, FDSpecifier;
    {
      ExecutorPath = std::make_unique<char[]>(OutOfProcessExecutor.size() + 1);
      strcpy(ExecutorPath.get(), OutOfProcessExecutor.data());

      std::string FDSpecifierStr("filedescs=");
      FDSpecifierStr += llvm::utostr(ToExecutor[ReadEnd]);
      FDSpecifierStr += ',';
      FDSpecifierStr += llvm::utostr(FromExecutor[WriteEnd]);
      FDSpecifier = std::make_unique<char[]>(FDSpecifierStr.size() + 1);
      strcpy(FDSpecifier.get(), FDSpecifierStr.c_str());
    }

    char *const Args[] = {ExecutorPath.get(), FDSpecifier.get(), nullptr};
    int RC = execvp(ExecutorPath.get(), Args);
    if (RC != 0) {
      llvm::errs() << "unable to launch out-of-process executor \""
                   << ExecutorPath.get() << "\"\n";
      exit(1);
    }
  }
  // else we're the parent...

  // Close the child ends of the pipes
  close(ToExecutor[ReadEnd]);
  close(FromExecutor[WriteEnd]);

  auto S = llvm::orc::SimpleRemoteEPC::Setup();

  return llvm::orc::SimpleRemoteEPC::Create<
      llvm::orc::FDSimpleRemoteEPCTransport>(
      std::make_unique<llvm::orc::DynamicThreadPoolTaskDispatcher>(),
      std::move(S), FromExecutor[ReadEnd], ToExecutor[WriteEnd]);
#endif
}

#if LLVM_ON_UNIX && LLVM_ENABLE_THREADS
static llvm::Error createTCPSocketError(llvm::Twine Details) {
  return llvm::make_error<llvm::StringError>(
      formatv("Failed to connect TCP socket '{0}': {1}",
              OutOfProcessExecutorConnect, Details),
      llvm::inconvertibleErrorCode());
}

static llvm::Expected<int> connectTCPSocket(std::string Host,
                                            std::string PortStr) {
  addrinfo *AI;
  addrinfo Hints{};
  Hints.ai_family = AF_INET;
  Hints.ai_socktype = SOCK_STREAM;
  Hints.ai_flags = AI_NUMERICSERV;

  if (int EC = getaddrinfo(Host.c_str(), PortStr.c_str(), &Hints, &AI))
    return createTCPSocketError("Address resolution failed (" +
                                llvm::StringRef(gai_strerror(EC)) + ")");

  // Cycle through the returned addrinfo structures and connect to the first
  // reachable endpoint.
  int SockFD;
  addrinfo *Server;
  for (Server = AI; Server != nullptr; Server = Server->ai_next) {
    // socket might fail, e.g. if the address family is not supported. Skip to
    // the next addrinfo structure in such a case.
    if ((SockFD = socket(AI->ai_family, AI->ai_socktype, AI->ai_protocol)) < 0)
      continue;

    // If connect returns null, we exit the loop with a working socket.
    if (connect(SockFD, Server->ai_addr, Server->ai_addrlen) == 0)
      break;

    close(SockFD);
  }
  freeaddrinfo(AI);

  // If we reached the end of the loop without connecting to a valid endpoint,
  // dump the last error that was logged in socket() or connect().
  if (Server == nullptr)
    return createTCPSocketError(std::strerror(errno));

  return SockFD;
}
#endif

static llvm::Expected<std::unique_ptr<llvm::orc::ExecutorProcessControl>>
connectToExecutor() {
#ifndef LLVM_ON_UNIX
  // FIXME: Add TCP support for Windows.
  return llvm::make_error<StringError>(
      "-" + OutOfProcessExecutorConnect.ArgStr +
          " not supported on non-unix platforms",
      inconvertibleErrorCode());
#elif !LLVM_ENABLE_THREADS
  // Out of process mode using SimpleRemoteEPC depends on threads.
  return llvm::make_error<StringError>(
      "-" + OutOfProcessExecutorConnect.ArgStr +
          " requires threads, but LLVM was built with "
          "LLVM_ENABLE_THREADS=Off",
      inconvertibleErrorCode());
#else

  llvm::StringRef Host, PortStr;
  std::tie(Host, PortStr) =
      llvm::StringRef(OutOfProcessExecutorConnect).split(':');
  if (Host.empty())
    return createTCPSocketError("Host name for -" +
                                OutOfProcessExecutorConnect.ArgStr +
                                " can not be empty");
  if (PortStr.empty())
    return createTCPSocketError("Port number in -" +
                                OutOfProcessExecutorConnect.ArgStr +
                                " can not be empty");
  int Port = 0;
  if (PortStr.getAsInteger(10, Port))
    return createTCPSocketError("Port number '" + PortStr +
                                "' is not a valid integer");

  llvm::Expected<int> SockFD = connectTCPSocket(Host.str(), PortStr.str());
  if (!SockFD)
    return SockFD.takeError();

  auto S = llvm::orc::SimpleRemoteEPC::Setup();

  return llvm::orc::SimpleRemoteEPC::Create<
      llvm::orc::FDSimpleRemoteEPCTransport>(
      std::make_unique<llvm::orc::DynamicThreadPoolTaskDispatcher>(),
      std::move(S), *SockFD, *SockFD);
#endif
}

llvm::ExitOnError ExitOnErr;
int main(int argc, const char **argv) {
  ExitOnErr.setBanner("clang-repl: ");
  llvm::cl::ParseCommandLineOptions(argc, argv);

  llvm::llvm_shutdown_obj Y; // Call llvm_shutdown() on exit.

  std::vector<const char *> ClangArgv(ClangArgs.size());
  std::transform(ClangArgs.begin(), ClangArgs.end(), ClangArgv.begin(),
                 [](const std::string &s) -> const char * { return s.data(); });
  // Initialize all targets (required for device offloading)
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();

  if (OptHostSupportsJit) {
    auto J = llvm::orc::LLJITBuilder().create();
    if (J)
      llvm::outs() << "true\n";
    else {
      llvm::consumeError(J.takeError());
      llvm::outs() << "false\n";
    }
    return 0;
  }

  clang::IncrementalCompilerBuilder CB;
  CB.SetCompilerArgs(ClangArgv);

  std::unique_ptr<clang::CompilerInstance> DeviceCI;
  if (CudaEnabled) {
    if (!CudaPath.empty())
      CB.SetCudaSDK(CudaPath);

    if (OffloadArch.empty()) {
      OffloadArch = "sm_35";
    }
    CB.SetOffloadArch(OffloadArch);

    DeviceCI = ExitOnErr(CB.CreateCudaDevice());
  }

  // FIXME: Investigate if we could use runToolOnCodeWithArgs from tooling. It
  // can replace the boilerplate code for creation of the compiler instance.
  std::unique_ptr<clang::CompilerInstance> CI;
  if (CudaEnabled) {
    CI = ExitOnErr(CB.CreateCudaHost());
  } else {
    CI = ExitOnErr(CB.CreateCpp());
  }

  // Set an error handler, so that any LLVM backend diagnostics go through our
  // error handler.
  llvm::install_fatal_error_handler(LLVMErrorHandler,
                                    static_cast<void *>(&CI->getDiagnostics()));

  // Load any requested plugins.
  CI->LoadRequestedPlugins();
  if (CudaEnabled)
    DeviceCI->LoadRequestedPlugins();

  ExitOnErr(sanitizeOopArguments(argv[0]));

  std::unique_ptr<clang::Interpreter> Interp;

  if (CudaEnabled) {
    Interp = ExitOnErr(
        clang::Interpreter::createWithCUDA(std::move(CI), std::move(DeviceCI)));

    if (CudaPath.empty()) {
      ExitOnErr(Interp->LoadDynamicLibrary("libcudart.so"));
    } else {
      auto CudaRuntimeLibPath = CudaPath + "/lib/libcudart.so";
      ExitOnErr(Interp->LoadDynamicLibrary(CudaRuntimeLibPath.c_str()));
    }
  } else if (OutOfProcessExecutor.getNumOccurrences()) {
    // Create an instance of llvm-jitlink-executor in a separate process.
    auto oopExecutor = ExitOnErr(launchExecutor());
    Interp = ExitOnErr(clang::Interpreter::createWithOutOfProcessExecutor(
        std::move(CI), std::move(oopExecutor), OrcRuntimePath));
  } else if (OutOfProcessExecutorConnect.getNumOccurrences()) {
    /// If -oop-executor-connect is passed then connect to the executor.
    auto REPC = ExitOnErr(connectToExecutor());
    Interp = ExitOnErr(clang::Interpreter::createWithOutOfProcessExecutor(
        std::move(CI), std::move(REPC), OrcRuntimePath));
  } else
    Interp = ExitOnErr(clang::Interpreter::create(std::move(CI)));

  for (const std::string &input : OptInputs) {
    if (auto Err = Interp->ParseAndExecute(input))
      llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(), "error: ");
  }

  bool HasError = false;

  if (OptInputs.empty()) {
    llvm::LineEditor LE("clang-repl");
    std::string Input;
    LE.setListCompleter(ReplListCompleter(CB, *Interp));
    while (std::optional<std::string> Line = LE.readLine()) {
      llvm::StringRef L = *Line;
      L = L.trim();
      if (L.endswith("\\")) {
        // FIXME: Support #ifdef X \ ...
        Input += L.drop_back(1);
        LE.setPrompt("clang-repl...   ");
        continue;
      }

      Input += L;
      if (Input == R"(%quit)") {
        break;
      }
      if (Input == R"(%undo)") {
        if (auto Err = Interp->Undo()) {
          llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(), "error: ");
          HasError = true;
        }
      } else if (Input.rfind("%lib ", 0) == 0) {
        if (auto Err = Interp->LoadDynamicLibrary(Input.data() + 5)) {
          llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(), "error: ");
          HasError = true;
        }
      } else if (auto Err = Interp->ParseAndExecute(Input)) {
        llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(), "error: ");
        HasError = true;
      }

      Input = "";
      LE.setPrompt("clang-repl> ");
    }
  }

  // Our error handler depends on the Diagnostics object, which we're
  // potentially about to delete. Uninstall the handler now so that any
  // later errors use the default handling behavior instead.
  llvm::remove_fatal_error_handler();

  return checkDiagErrors(Interp->getCompilerInstance(), HasError);
}
