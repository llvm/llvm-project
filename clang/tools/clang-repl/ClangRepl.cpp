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

#include "clang/Interpreter/RemoteJITUtils.h"

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/Version.h"
#include "clang/Config/config.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Interpreter/CodeCompletion.h"
#include "clang/Interpreter/Interpreter.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Sema.h"

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/LineEditor/LineEditor.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h" // llvm_shutdown
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"
#include <optional>

#include "llvm/ExecutionEngine/Orc/Debugging/DebuggerSupport.h"

// Disable LSan for this test.
// FIXME: Re-enable once we can assume GCC 13.2 or higher.
// https://llvm.org/github.com/llvm/llvm-project/issues/67586.
#if LLVM_ADDRESS_SANITIZER_BUILD || LLVM_HWADDRESS_SANITIZER_BUILD
#include <sanitizer/lsan_interface.h>
LLVM_ATTRIBUTE_USED int __lsan_is_turned_off() { return 1; }
#endif

#define DEBUG_TYPE "clang-repl"

static llvm::cl::opt<bool> CudaEnabled("cuda", llvm::cl::Hidden);
static llvm::cl::opt<std::string> CudaPath("cuda-path", llvm::cl::Hidden);
static llvm::cl::opt<std::string> OffloadArch("offload-arch", llvm::cl::Hidden);
static llvm::cl::OptionCategory OOPCategory("Out-of-process Execution Options");
static llvm::cl::opt<std::string> SlabAllocateSizeString(
    "slab-allocate",
    llvm::cl::desc("Allocate from a slab of the given size "
                   "(allowable suffixes: Kb, Mb, Gb. default = "
                   "Kb)"),
    llvm::cl::init(""), llvm::cl::cat(OOPCategory));
static llvm::cl::opt<std::string>
    OOPExecutor("oop-executor",
                llvm::cl::desc("Launch an out-of-process executor to run code"),
                llvm::cl::init(""), llvm::cl::ValueOptional,
                llvm::cl::cat(OOPCategory));
static llvm::cl::opt<std::string> OOPExecutorConnect(
    "oop-executor-connect",
    llvm::cl::desc(
        "Connect to an out-of-process executor through a TCP socket"),
    llvm::cl::value_desc("<hostname>:<port>"));
static llvm::cl::opt<std::string>
    OrcRuntimePath("orc-runtime", llvm::cl::desc("Path to the ORC runtime"),
                   llvm::cl::init(""), llvm::cl::ValueOptional,
                   llvm::cl::cat(OOPCategory));
static llvm::cl::opt<bool> UseSharedMemory(
    "use-shared-memory",
    llvm::cl::desc("Use shared memory to transfer generated code and data"),
    llvm::cl::init(false), llvm::cl::cat(OOPCategory));
static llvm::cl::list<std::string>
    ClangArgs("Xcc",
              llvm::cl::desc("Argument to pass to the CompilerInvocation"),
              llvm::cl::CommaSeparated);
static llvm::cl::opt<bool> OptHostSupportsJit("host-supports-jit",
                                              llvm::cl::Hidden);
static llvm::cl::list<std::string> OptInputs(llvm::cl::Positional,
                                             llvm::cl::desc("[code to run]"));

static llvm::Error sanitizeOopArguments(const char *ArgV0) {
  // Only one of -oop-executor and -oop-executor-connect can be used.
  if (!!OOPExecutor.getNumOccurrences() &&
      !!OOPExecutorConnect.getNumOccurrences())
    return llvm::make_error<llvm::StringError>(
        "Only one of -" + OOPExecutor.ArgStr + " and -" +
            OOPExecutorConnect.ArgStr + " can be specified",
        llvm::inconvertibleErrorCode());

  llvm::Triple SystemTriple(llvm::sys::getProcessTriple());
  // TODO: Remove once out-of-process execution support is implemented for
  // non-Unix platforms.
  if ((!SystemTriple.isOSBinFormatELF() &&
       !SystemTriple.isOSBinFormatMachO()) &&
      (OOPExecutor.getNumOccurrences() ||
       OOPExecutorConnect.getNumOccurrences()))
    return llvm::make_error<llvm::StringError>(
        "Out-of-process execution is only supported on Unix platforms",
        llvm::inconvertibleErrorCode());

  // If -slab-allocate is passed, check that we're not trying to use it in
  // -oop-executor or -oop-executor-connect mode.
  //
  // FIXME: Remove once we enable remote slab allocation.
  if (SlabAllocateSizeString != "") {
    if (OOPExecutor.getNumOccurrences() ||
        OOPExecutorConnect.getNumOccurrences())
      return llvm::make_error<llvm::StringError>(
          "-slab-allocate cannot be used with -oop-executor or "
          "-oop-executor-connect",
          llvm::inconvertibleErrorCode());
  }

  // Out-of-process executors require the ORC runtime.
  if (OrcRuntimePath.empty() && (OOPExecutor.getNumOccurrences() ||
                                 OOPExecutorConnect.getNumOccurrences())) {
    llvm::SmallString<256> BasePath(llvm::sys::fs::getMainExecutable(
        ArgV0, reinterpret_cast<void *>(&sanitizeOopArguments)));
    llvm::sys::path::remove_filename(BasePath); // Remove clang-repl filename.
    llvm::sys::path::remove_filename(BasePath); // Remove ./bin directory.
    llvm::sys::path::append(BasePath, CLANG_INSTALL_LIBDIR_BASENAME, "clang",
                            CLANG_VERSION_MAJOR_STRING);
    if (SystemTriple.isOSBinFormatELF())
      OrcRuntimePath =
          BasePath.str().str() + "lib/x86_64-unknown-linux-gnu/liborc_rt.a";
    else if (SystemTriple.isOSBinFormatMachO())
      OrcRuntimePath = BasePath.str().str() + "/lib/darwin/liborc_rt_osx.a";
    else
      return llvm::make_error<llvm::StringError>(
          "Out-of-process execution is not supported on non-unix platforms",
          llvm::inconvertibleErrorCode());
  }

  // If -oop-executor was used but no value was specified then use a sensible
  // default.
  if (!!OOPExecutor.getNumOccurrences() && OOPExecutor.empty()) {
    llvm::SmallString<256> OOPExecutorPath(llvm::sys::fs::getMainExecutable(
        ArgV0, reinterpret_cast<void *>(&sanitizeOopArguments)));
    llvm::sys::path::remove_filename(OOPExecutorPath);
    llvm::sys::path::append(OOPExecutorPath, "llvm-jitlink-executor");
    OOPExecutor = OOPExecutorPath.str().str();
  }

  return llvm::Error::success();
}

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
  auto *MainCI = (*Interp)->getCompilerInstance();
  auto CC = clang::ReplCodeCompleter();
  CC.codeComplete(MainCI, Buffer, Lines, Pos + 1,
                  MainInterp.getCompilerInstance(), Results);
  for (auto c : Results) {
    if (c.find(CC.Prefix) == 0)
      Comps.push_back(
          llvm::LineEditor::Completion(c.substr(CC.Prefix.size()), c));
  }
  return Comps;
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
  llvm::InitializeAllAsmParsers();

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

  ExitOnErr(sanitizeOopArguments(argv[0]));

  std::unique_ptr<llvm::orc::ExecutorProcessControl> EPC;
  if (OOPExecutor.getNumOccurrences()) {
    // Launch an out-of-process executor locally in a child process.
    EPC = ExitOnErr(
        launchExecutor(OOPExecutor, UseSharedMemory, SlabAllocateSizeString));
  } else if (OOPExecutorConnect.getNumOccurrences()) {
    EPC = ExitOnErr(connectTCPSocket(OOPExecutorConnect, UseSharedMemory,
                                     SlabAllocateSizeString));
  }

  std::unique_ptr<llvm::orc::LLJITBuilder> JB;
  if (EPC) {
    CB.SetTargetTriple(EPC->getTargetTriple().getTriple());
    JB = ExitOnErr(
        clang::Interpreter::createLLJITBuilder(std::move(EPC), OrcRuntimePath));
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
  } else if (JB) {
    Interp =
        ExitOnErr(clang::Interpreter::create(std::move(CI), std::move(JB)));
  } else
    Interp = ExitOnErr(clang::Interpreter::create(std::move(CI)));

  bool HasError = false;

  for (const std::string &input : OptInputs) {
    if (auto Err = Interp->ParseAndExecute(input)) {
      llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(), "error: ");
      HasError = true;
    }
  }

  if (OptInputs.empty()) {
    llvm::LineEditor LE("clang-repl");
    std::string Input;
    LE.setListCompleter(ReplListCompleter(CB, *Interp));
    while (std::optional<std::string> Line = LE.readLine()) {
      llvm::StringRef L = *Line;
      L = L.trim();
      if (L.ends_with("\\")) {
        Input += L.drop_back(1);
        // If it is a preprocessor directive, new lines matter.
        if (L.starts_with('#'))
          Input += "\n";
        LE.setPrompt("clang-repl...   ");
        continue;
      }

      Input += L;
      if (Input == R"(%quit)") {
        break;
      }
      if (Input == R"(%undo)") {
        if (auto Err = Interp->Undo())
          llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(), "error: ");
      } else if (Input.rfind("%lib ", 0) == 0) {
        if (auto Err = Interp->LoadDynamicLibrary(Input.data() + 5))
          llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(), "error: ");
      } else if (auto Err = Interp->ParseAndExecute(Input)) {
        llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(), "error: ");
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
