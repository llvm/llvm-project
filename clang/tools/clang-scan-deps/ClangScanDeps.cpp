//===- ClangScanDeps.cpp - Implementation of clang-scan-deps --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/CommonOptionsParser.h"
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

/// The high-level implementation of the dependency discovery tool that runs on
/// an individual worker thread.
class DependencyScanningTool {
public:
  /// Construct a dependency scanning tool.
  ///
  /// \param Compilations     The reference to the compilation database that's
  /// used by the clang tool.
  DependencyScanningTool(const tooling::CompilationDatabase &Compilations,
                         SharedStream &OS, SharedStream &Errs)
      : Compilations(Compilations), OS(OS), Errs(Errs) {}

  /// Computes the dependencies for the given file and prints them out.
  ///
  /// \returns True on error.
  bool runOnFile(const std::string &Input, StringRef CWD) {
    auto MaybeFile = Worker.getDependencyFile(Input, CWD, Compilations);
    if (!MaybeFile) {
      llvm::handleAllErrors(
          MaybeFile.takeError(), [this, &Input](llvm::StringError &Err) {
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

private:
  DependencyScanningWorker Worker;
  const tooling::CompilationDatabase &Compilations;
  SharedStream &OS;
  SharedStream &Errs;
};

llvm::cl::opt<bool> Help("h", llvm::cl::desc("Alias for -help"),
                         llvm::cl::Hidden);

llvm::cl::OptionCategory DependencyScannerCategory("Tool options");

llvm::cl::opt<unsigned>
    NumThreads("j", llvm::cl::Optional,
               llvm::cl::desc("Number of worker threads to use (default: use "
                              "all concurrent threads)"),
               llvm::cl::init(0));

llvm::cl::opt<std::string>
    CompilationDB("compilation-database",
                  llvm::cl::desc("Compilation database"), llvm::cl::Required,
                  llvm::cl::cat(DependencyScannerCategory));

} // end anonymous namespace

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
      llvm::make_unique<tooling::ArgumentsAdjustingCompilations>(
          std::move(Compilations));
  AdjustingCompilations->appendArgumentsAdjuster(
      [](const tooling::CommandLineArguments &Args, StringRef /*unused*/) {
        tooling::CommandLineArguments AdjustedArgs = Args;
        AdjustedArgs.push_back("-o");
        AdjustedArgs.push_back("/dev/null");
        AdjustedArgs.push_back("-Xclang");
        AdjustedArgs.push_back("-Eonly");
        AdjustedArgs.push_back("-Xclang");
        AdjustedArgs.push_back("-sys-header-deps");
        AdjustedArgs.push_back("-Wno-error");
        return AdjustedArgs;
      });

  SharedStream Errs(llvm::errs());
  // Print out the dependency results to STDOUT by default.
  SharedStream DependencyOS(llvm::outs());
  unsigned NumWorkers =
      NumThreads == 0 ? llvm::hardware_concurrency() : NumThreads;
  std::vector<std::unique_ptr<DependencyScanningTool>> WorkerTools;
  for (unsigned I = 0; I < NumWorkers; ++I)
    WorkerTools.push_back(llvm::make_unique<DependencyScanningTool>(
        *AdjustingCompilations, DependencyOS, Errs));

  std::vector<std::thread> WorkerThreads;
  std::atomic<bool> HadErrors(false);
  std::mutex Lock;
  size_t Index = 0;

  llvm::outs() << "Running clang-scan-deps on " << Inputs.size()
               << " files using " << NumWorkers << " workers\n";
  for (unsigned I = 0; I < NumWorkers; ++I) {
    WorkerThreads.emplace_back(
        [I, &Lock, &Index, &Inputs, &HadErrors, &WorkerTools]() {
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
            if (WorkerTools[I]->runOnFile(Input, CWD))
              HadErrors = true;
          }
        });
  }
  for (auto &W : WorkerThreads)
    W.join();

  return HadErrors;
}
