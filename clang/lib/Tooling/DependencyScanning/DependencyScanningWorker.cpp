//===- DependencyScanningWorker.cpp - clang-scan-deps worker --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/DependencyScanning/DependencyScanningWorker.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/Utils.h"
#include "clang/Tooling/Tooling.h"

using namespace clang;
using namespace tooling;
using namespace dependencies;

namespace {

/// Prints out all of the gathered dependencies into a string.
class DependencyPrinter : public DependencyFileGenerator {
public:
  DependencyPrinter(std::unique_ptr<DependencyOutputOptions> Opts,
                    std::string &S)
      : DependencyFileGenerator(*Opts), Opts(std::move(Opts)), S(S) {}

  void finishedMainFile(DiagnosticsEngine &Diags) override {
    llvm::raw_string_ostream OS(S);
    outputDependencyFile(OS);
  }

private:
  std::unique_ptr<DependencyOutputOptions> Opts;
  std::string &S;
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

/// A clang tool that runs the preprocessor in a mode that's optimized for
/// dependency scanning for the given compiler invocation.
class DependencyScanningAction : public tooling::ToolAction {
public:
  DependencyScanningAction(StringRef WorkingDirectory,
                           std::string &DependencyFileContents)
      : WorkingDirectory(WorkingDirectory),
        DependencyFileContents(DependencyFileContents) {}

  bool runInvocation(std::shared_ptr<CompilerInvocation> Invocation,
                     FileManager *FileMgr,
                     std::shared_ptr<PCHContainerOperations> PCHContainerOps,
                     DiagnosticConsumer *DiagConsumer) override {
    // Create a compiler instance to handle the actual work.
    CompilerInstance Compiler(std::move(PCHContainerOps));
    Compiler.setInvocation(std::move(Invocation));
    FileMgr->getFileSystemOpts().WorkingDir = WorkingDirectory;
    Compiler.setFileManager(FileMgr);

    // Don't print 'X warnings and Y errors generated'.
    Compiler.getDiagnosticOpts().ShowCarets = false;
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
    Compiler.addDependencyCollector(std::make_shared<DependencyPrinter>(
        std::move(Opts), DependencyFileContents));

    auto Action = llvm::make_unique<PreprocessOnlyAction>();
    const bool Result = Compiler.ExecuteAction(*Action);
    FileMgr->clearStatCache();
    return Result;
  }

private:
  StringRef WorkingDirectory;
  /// The dependency file will be written to this string.
  std::string &DependencyFileContents;
};

} // end anonymous namespace

DependencyScanningWorker::DependencyScanningWorker() {
  DiagOpts = new DiagnosticOptions();
  PCHContainerOps = std::make_shared<PCHContainerOperations>();
  /// FIXME: Use the shared file system from the service for fast scanning
  /// mode.
  WorkerFS = new ProxyFileSystemWithoutChdir(llvm::vfs::getRealFileSystem());
}

llvm::Expected<std::string>
DependencyScanningWorker::getDependencyFile(const std::string &Input,
                                            StringRef WorkingDirectory,
                                            const CompilationDatabase &CDB) {
  // Capture the emitted diagnostics and report them to the client
  // in the case of a failure.
  std::string DiagnosticOutput;
  llvm::raw_string_ostream DiagnosticsOS(DiagnosticOutput);
  TextDiagnosticPrinter DiagPrinter(DiagnosticsOS, DiagOpts.get());

  WorkerFS->setCurrentWorkingDirectory(WorkingDirectory);
  tooling::ClangTool Tool(CDB, Input, PCHContainerOps, WorkerFS);
  Tool.clearArgumentsAdjusters();
  Tool.setRestoreWorkingDir(false);
  Tool.setPrintErrorMessage(false);
  Tool.setDiagnosticConsumer(&DiagPrinter);
  std::string Output;
  DependencyScanningAction Action(WorkingDirectory, Output);
  if (Tool.run(&Action)) {
    return llvm::make_error<llvm::StringError>(DiagnosticsOS.str(),
                                               llvm::inconvertibleErrorCode());
  }
  return Output;
}
