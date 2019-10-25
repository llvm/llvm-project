//===- ClangScanDeps.cpp - Implementation of clang-scan-deps --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningService.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningTool.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningWorker.h"
#include "clang/Tooling/JSONCompilationDatabase.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Options.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/Threading.h"
#include <mutex>
#include <thread>

using namespace clang;
using namespace tooling::dependencies;

namespace {

class SharedStream {
public:
  SharedStream(raw_ostream &OS) : OS(OS) {}
  void applyLocked(llvm::function_ref<void(raw_ostream &OS)> Fn) {
    std::unique_lock<std::mutex> LockGuard(Lock);
    Fn(OS);
    OS.flush();
  }

private:
  std::mutex Lock;
  raw_ostream &OS;
};

llvm::cl::opt<bool> Help("h", llvm::cl::desc("Alias for -help"),
                         llvm::cl::Hidden);

llvm::cl::OptionCategory DependencyScannerCategory("Tool options");

static llvm::cl::opt<ScanningMode> ScanMode(
    "mode",
    llvm::cl::desc("The preprocessing mode used to compute the dependencies"),
    llvm::cl::values(
        clEnumValN(ScanningMode::MinimizedSourcePreprocessing,
                   "preprocess-minimized-sources",
                   "The set of dependencies is computed by preprocessing the "
                   "source files that were minimized to only include the "
                   "contents that might affect the dependencies"),
        clEnumValN(ScanningMode::CanonicalPreprocessing, "preprocess",
                   "The set of dependencies is computed by preprocessing the "
                   "unmodified source files")),
    llvm::cl::init(ScanningMode::MinimizedSourcePreprocessing),
    llvm::cl::cat(DependencyScannerCategory));

llvm::cl::opt<unsigned>
    NumThreads("j", llvm::cl::Optional,
               llvm::cl::desc("Number of worker threads to use (default: use "
                              "all concurrent threads)"),
               llvm::cl::init(0), llvm::cl::cat(DependencyScannerCategory));

llvm::cl::opt<std::string>
    CompilationDB("compilation-database",
                  llvm::cl::desc("Compilation database"), llvm::cl::Required,
                  llvm::cl::cat(DependencyScannerCategory));

llvm::cl::opt<bool> ReuseFileManager(
    "reuse-filemanager",
    llvm::cl::desc("Reuse the file manager and its cache between invocations."),
    llvm::cl::init(true), llvm::cl::cat(DependencyScannerCategory));

llvm::cl::opt<bool> SkipExcludedPPRanges(
    "skip-excluded-pp-ranges",
    llvm::cl::desc(
        "Use the preprocessor optimization that skips excluded conditionals by "
        "bumping the buffer pointer in the lexer instead of lexing the tokens  "
        "until reaching the end directive."),
    llvm::cl::init(true), llvm::cl::cat(DependencyScannerCategory));

llvm::cl::opt<bool> Verbose("v", llvm::cl::Optional,
                            llvm::cl::desc("Use verbose output."),
                            llvm::cl::init(false),
                            llvm::cl::cat(DependencyScannerCategory));

} // end anonymous namespace

/// \returns object-file path derived from source-file path.
static std::string getObjFilePath(StringRef SrcFile) {
  SmallString<128> ObjFileName(SrcFile);
  llvm::sys::path::replace_extension(ObjFileName, "o");
  return ObjFileName.str();
}

/// Takes the result of a dependency scan and prints error / dependency files
/// based on the result.
///
/// \returns True on error.
static bool handleDependencyToolResult(const std::string &Input,
                                       llvm::Expected<std::string> &MaybeFile,
                                       SharedStream &OS, SharedStream &Errs) {
  if (!MaybeFile) {
    llvm::handleAllErrors(
        MaybeFile.takeError(), [&Input, &Errs](llvm::StringError &Err) {
          Errs.applyLocked([&](raw_ostream &OS) {
            OS << "Error while scanning dependencies for " << Input << ":\n";
            OS << Err.getMessage();
          });
        });
    return true;
  }
  OS.applyLocked([&](raw_ostream &OS) { OS << *MaybeFile; });
  return false;
}

int main(int argc, const char **argv) {
  llvm::InitLLVM X(argc, argv);
  llvm::cl::HideUnrelatedOptions(DependencyScannerCategory);
  if (!llvm::cl::ParseCommandLineOptions(argc, argv))
    return 1;

  std::string ErrorMessage;
  std::unique_ptr<tooling::JSONCompilationDatabase> Compilations =
      tooling::JSONCompilationDatabase::loadFromFile(
          CompilationDB, ErrorMessage,
          tooling::JSONCommandLineSyntax::AutoDetect);
  if (!Compilations) {
    llvm::errs() << "error: " << ErrorMessage << "\n";
    return 1;
  }

  llvm::cl::PrintOptionValues();

  // By default the tool runs on all inputs in the CDB.
  std::vector<std::pair<std::string, std::string>> Inputs;
  for (const auto &Command : Compilations->getAllCompileCommands())
    Inputs.emplace_back(Command.Filename, Command.Directory);

  // The command options are rewritten to run Clang in preprocessor only mode.
  auto AdjustingCompilations =
      std::make_unique<tooling::ArgumentsAdjustingCompilations>(
          std::move(Compilations));
  AdjustingCompilations->appendArgumentsAdjuster(
      [](const tooling::CommandLineArguments &Args, StringRef FileName) {
        std::string LastO = "";
        bool HasMT = false;
        bool HasMQ = false;
        bool HasMD = false;
        // We need to find the last -o value.
        if (!Args.empty()) {
          std::size_t Idx = Args.size() - 1;
          for (auto It = Args.rbegin(); It != Args.rend(); ++It) {
            if (It != Args.rbegin()) {
              if (Args[Idx] == "-o")
                LastO = Args[Idx + 1];
              if (Args[Idx] == "-MT")
                HasMT = true;
              if (Args[Idx] == "-MQ")
                HasMQ = true;
              if (Args[Idx] == "-MD")
                HasMD = true;
            }
            --Idx;
          }
        }
        // If there's no -MT/-MQ Driver would add -MT with the value of the last
        // -o option.
        tooling::CommandLineArguments AdjustedArgs = Args;
        AdjustedArgs.push_back("-o");
        AdjustedArgs.push_back("/dev/null");
        if (!HasMT && !HasMQ) {
          AdjustedArgs.push_back("-M");
          AdjustedArgs.push_back("-MT");
          // We're interested in source dependencies of an object file.
          if (!HasMD) {
            // FIXME: We are missing the directory unless the -o value is an
            // absolute path.
            AdjustedArgs.push_back(!LastO.empty() ? LastO
                                                  : getObjFilePath(FileName));
          } else {
            AdjustedArgs.push_back(FileName);
          }
        }
        AdjustedArgs.push_back("-Xclang");
        AdjustedArgs.push_back("-Eonly");
        AdjustedArgs.push_back("-Xclang");
        AdjustedArgs.push_back("-sys-header-deps");
        AdjustedArgs.push_back("-Wno-error");
        return AdjustedArgs;
      });
  AdjustingCompilations->appendArgumentsAdjuster(
      tooling::getClangStripSerializeDiagnosticAdjuster());

  SharedStream Errs(llvm::errs());
  // Print out the dependency results to STDOUT by default.
  SharedStream DependencyOS(llvm::outs());

  DependencyScanningService Service(ScanMode, ReuseFileManager,
                                    SkipExcludedPPRanges);
#if LLVM_ENABLE_THREADS
  unsigned NumWorkers =
      NumThreads == 0 ? llvm::hardware_concurrency() : NumThreads;
#else
  unsigned NumWorkers = 1;
#endif
  std::vector<std::unique_ptr<DependencyScanningTool>> WorkerTools;
  for (unsigned I = 0; I < NumWorkers; ++I)
    WorkerTools.push_back(std::make_unique<DependencyScanningTool>(
        Service, *AdjustingCompilations));

  std::vector<std::thread> WorkerThreads;
  std::atomic<bool> HadErrors(false);
  std::mutex Lock;
  size_t Index = 0;

  if (Verbose) {
    llvm::outs() << "Running clang-scan-deps on " << Inputs.size()
                 << " files using " << NumWorkers << " workers\n";
  }
  for (unsigned I = 0; I < NumWorkers; ++I) {
    auto Worker = [I, &Lock, &Index, &Inputs, &HadErrors, &WorkerTools,
                   &DependencyOS, &Errs]() {
      while (true) {
        std::string Input;
        StringRef CWD;
        // Take the next input.
        {
          std::unique_lock<std::mutex> LockGuard(Lock);
          if (Index >= Inputs.size())
            return;
          const auto &Compilation = Inputs[Index++];
          Input = Compilation.first;
          CWD = Compilation.second;
        }
        // Run the tool on it.
        auto MaybeFile = WorkerTools[I]->getDependencyFile(Input, CWD);
        if (handleDependencyToolResult(Input, MaybeFile, DependencyOS, Errs))
          HadErrors = true;
      }
    };
#if LLVM_ENABLE_THREADS
    WorkerThreads.emplace_back(std::move(Worker));
#else
    // Run the worker without spawning a thread when threads are disabled.
    Worker();
#endif
  }
  for (auto &W : WorkerThreads)
    W.join();

  return HadErrors;
}
