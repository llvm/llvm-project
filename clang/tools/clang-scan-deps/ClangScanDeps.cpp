//===- ClangScanDeps.cpp - Implementation of clang-scan-deps --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/PCHContainerOperations.h"
#include "clang/FrontendTool/Utils.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/JSONCompilationDatabase.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Options.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/Threading.h"
#include <mutex>
#include <thread>

using namespace clang;

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

/// Prints out all of the gathered dependencies into one output stream instead
/// of using the output dependency file.
class DependencyPrinter : public DependencyFileGenerator {
public:
  DependencyPrinter(std::unique_ptr<DependencyOutputOptions> Opts,
                    SharedStream &OS)
      : DependencyFileGenerator(*Opts), Opts(std::move(Opts)), OS(OS) {}

  void finishedMainFile(DiagnosticsEngine &Diags) override {
    OS.applyLocked([this](raw_ostream &OS) { outputDependencyFile(OS); });
  }

private:
  std::unique_ptr<DependencyOutputOptions> Opts;
  SharedStream &OS;
};

/// A clang tool that runs the preprocessor only for the given compiler
/// invocation.
class PreprocessorOnlyTool : public tooling::ToolAction {
public:
  PreprocessorOnlyTool(StringRef WorkingDirectory, SharedStream &OS)
      : WorkingDirectory(WorkingDirectory), OS(OS) {}

  bool runInvocation(std::shared_ptr<CompilerInvocation> Invocation,
                     FileManager *FileMgr,
                     std::shared_ptr<PCHContainerOperations> PCHContainerOps,
                     DiagnosticConsumer *DiagConsumer) override {
    // Create a compiler instance to handle the actual work.
    CompilerInstance Compiler(std::move(PCHContainerOps));
    Compiler.setInvocation(std::move(Invocation));
    FileMgr->getFileSystemOpts().WorkingDir = WorkingDirectory;
    Compiler.setFileManager(FileMgr);

    // Create the compiler's actual diagnostics engine.
    Compiler.createDiagnostics(DiagConsumer, /*ShouldOwnClient=*/false);
    if (!Compiler.hasDiagnostics())
      return false;

    Compiler.createSourceManager(*FileMgr);

    // Create the dependency collector that will collect the produced
    // dependencies.
    //
    // This also moves the existing dependency output options from the
    // invocation to the collector. The options in the invocation are reset,
    // which ensures that the compiler won't create new dependency collectors,
    // and thus won't write out the extra '.d' files to disk.
    auto Opts = llvm::make_unique<DependencyOutputOptions>(
        std::move(Compiler.getInvocation().getDependencyOutputOpts()));
    // We need at least one -MT equivalent for the generator to work.
    if (Opts->Targets.empty())
      Opts->Targets = {"clang-scan-deps dependency"};
    Compiler.addDependencyCollector(
        std::make_shared<DependencyPrinter>(std::move(Opts), OS));

    auto Action = llvm::make_unique<PreprocessOnlyAction>();
    const bool Result = Compiler.ExecuteAction(*Action);
    FileMgr->clearStatCache();
    return Result;
  }

private:
  StringRef WorkingDirectory;
  SharedStream &OS;
};

/// A proxy file system that doesn't call `chdir` when changing the working
/// directory of a clang tool.
class ProxyFileSystemWithoutChdir : public llvm::vfs::ProxyFileSystem {
public:
  ProxyFileSystemWithoutChdir(
      llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS)
      : ProxyFileSystem(std::move(FS)) {}

  llvm::ErrorOr<std::string> getCurrentWorkingDirectory() const override {
    assert(!CWD.empty() && "empty CWD");
    return CWD;
  }

  std::error_code setCurrentWorkingDirectory(const Twine &Path) override {
    CWD = Path.str();
    return {};
  }

private:
  std::string CWD;
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
                         SharedStream &OS)
      : Compilations(Compilations), OS(OS) {
    PCHContainerOps = std::make_shared<PCHContainerOperations>();
    BaseFS = new ProxyFileSystemWithoutChdir(llvm::vfs::getRealFileSystem());
  }

  /// Computes the dependencies for the given file.
  ///
  /// \returns True on error.
  bool runOnFile(const std::string &Input, StringRef CWD) {
    BaseFS->setCurrentWorkingDirectory(CWD);
    tooling::ClangTool Tool(Compilations, Input, PCHContainerOps, BaseFS);
    Tool.clearArgumentsAdjusters();
    Tool.setRestoreWorkingDir(false);
    PreprocessorOnlyTool Action(CWD, OS);
    return Tool.run(&Action);
  }

private:
  const tooling::CompilationDatabase &Compilations;
  SharedStream &OS;
  std::shared_ptr<PCHContainerOperations> PCHContainerOps;
  /// The real filesystem used as a base for all the operations performed by the
  /// tool.
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> BaseFS;
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
        return AdjustedArgs;
      });

  // Print out the dependency results to STDOUT by default.
  SharedStream DependencyOS(llvm::outs());
  unsigned NumWorkers =
      NumThreads == 0 ? llvm::hardware_concurrency() : NumThreads;
  std::vector<std::unique_ptr<DependencyScanningTool>> WorkerTools;
  for (unsigned I = 0; I < NumWorkers; ++I)
    WorkerTools.push_back(llvm::make_unique<DependencyScanningTool>(
        *AdjustingCompilations, DependencyOS));

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
