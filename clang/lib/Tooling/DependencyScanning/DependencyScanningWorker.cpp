//===- DependencyScanningWorker.cpp - clang-scan-deps worker --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/DependencyScanning/DependencyScanningWorker.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/Utils.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningService.h"
#include "clang/Tooling/DependencyScanning/ModuleDepCollector.h"
#include "clang/Tooling/Tooling.h"

using namespace clang;
using namespace tooling;
using namespace dependencies;

namespace {

/// Forwards the gatherered dependencies to the consumer.
class DependencyConsumerForwarder : public DependencyFileGenerator {
public:
  DependencyConsumerForwarder(std::unique_ptr<DependencyOutputOptions> Opts,
                              DependencyConsumer &C)
      : DependencyFileGenerator(*Opts), Opts(std::move(Opts)), C(C) {}

  void finishedMainFile(DiagnosticsEngine &Diags) override {
    llvm::SmallString<256> CanonPath;
    for (const auto &File : getDependencies()) {
      CanonPath = File;
      llvm::sys::path::remove_dots(CanonPath, /*remove_dot_dot=*/true);
      C.handleFileDependency(*Opts, CanonPath);
    }
  }

private:
  std::unique_ptr<DependencyOutputOptions> Opts;
  DependencyConsumer &C;
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
  DependencyScanningAction(
      StringRef WorkingDirectory, DependencyConsumer &Consumer,
      llvm::IntrusiveRefCntPtr<DependencyScanningWorkerFilesystem> DepFS,
      ExcludedPreprocessorDirectiveSkipMapping *PPSkipMappings,
      ScanningOutputFormat Format)
      : WorkingDirectory(WorkingDirectory), Consumer(Consumer),
        DepFS(std::move(DepFS)), PPSkipMappings(PPSkipMappings),
        Format(Format) {}

  bool runInvocation(std::shared_ptr<CompilerInvocation> Invocation,
                     FileManager *FileMgr,
                     std::shared_ptr<PCHContainerOperations> PCHContainerOps,
                     DiagnosticConsumer *DiagConsumer) override {
    // Create a compiler instance to handle the actual work.
    CompilerInstance Compiler(std::move(PCHContainerOps));
    Compiler.setInvocation(std::move(Invocation));

    // Don't print 'X warnings and Y errors generated'.
    Compiler.getDiagnosticOpts().ShowCarets = false;
    // Create the compiler's actual diagnostics engine.
    Compiler.createDiagnostics(DiagConsumer, /*ShouldOwnClient=*/false);
    if (!Compiler.hasDiagnostics())
      return false;

    // Use the dependency scanning optimized file system if we can.
    if (DepFS) {
      const CompilerInvocation &CI = Compiler.getInvocation();
      // Add any filenames that were explicity passed in the build settings and
      // that might be opened, as we want to ensure we don't run source
      // minimization on them.
      DepFS->IgnoredFiles.clear();
      for (const auto &Entry : CI.getHeaderSearchOpts().UserEntries)
        DepFS->IgnoredFiles.insert(Entry.Path);
      for (const auto &Entry : CI.getHeaderSearchOpts().VFSOverlayFiles)
        DepFS->IgnoredFiles.insert(Entry);

      // Support for virtual file system overlays on top of the caching
      // filesystem.
      FileMgr->setVirtualFileSystem(createVFSFromCompilerInvocation(
          CI, Compiler.getDiagnostics(), DepFS));

      // Pass the skip mappings which should speed up excluded conditional block
      // skipping in the preprocessor.
      if (PPSkipMappings)
        Compiler.getPreprocessorOpts()
            .ExcludedConditionalDirectiveSkipMappings = PPSkipMappings;
    }

    FileMgr->getFileSystemOpts().WorkingDir = WorkingDirectory;
    Compiler.setFileManager(FileMgr);
    Compiler.createSourceManager(*FileMgr);

    // Create the dependency collector that will collect the produced
    // dependencies.
    //
    // This also moves the existing dependency output options from the
    // invocation to the collector. The options in the invocation are reset,
    // which ensures that the compiler won't create new dependency collectors,
    // and thus won't write out the extra '.d' files to disk.
    auto Opts = std::make_unique<DependencyOutputOptions>(
        std::move(Compiler.getInvocation().getDependencyOutputOpts()));
    // We need at least one -MT equivalent for the generator to work.
    if (Opts->Targets.empty())
      Opts->Targets = {"clang-scan-deps dependency"};

    switch (Format) {
    case ScanningOutputFormat::Make:
      Compiler.addDependencyCollector(
          std::make_shared<DependencyConsumerForwarder>(std::move(Opts),
                                                        Consumer));
      break;
    case ScanningOutputFormat::Full:
      Compiler.addDependencyCollector(
          std::make_shared<ModuleDepCollector>(Compiler, Consumer));
      break;
    }

    Consumer.handleContextHash(Compiler.getInvocation().getModuleHash());

    auto Action = std::make_unique<PreprocessOnlyAction>();
    const bool Result = Compiler.ExecuteAction(*Action);
    if (!DepFS)
      FileMgr->clearStatCache();
    return Result;
  }

private:
  StringRef WorkingDirectory;
  DependencyConsumer &Consumer;
  llvm::IntrusiveRefCntPtr<DependencyScanningWorkerFilesystem> DepFS;
  ExcludedPreprocessorDirectiveSkipMapping *PPSkipMappings;
  ScanningOutputFormat Format;
};

} // end anonymous namespace

DependencyScanningWorker::DependencyScanningWorker(
    DependencyScanningService &Service)
    : Format(Service.getFormat()) {
  DiagOpts = new DiagnosticOptions();
  PCHContainerOps = std::make_shared<PCHContainerOperations>();
  RealFS = new ProxyFileSystemWithoutChdir(llvm::vfs::getRealFileSystem());
  if (Service.canSkipExcludedPPRanges())
    PPSkipMappings =
        std::make_unique<ExcludedPreprocessorDirectiveSkipMapping>();
  if (Service.getMode() == ScanningMode::MinimizedSourcePreprocessing)
    DepFS = new DependencyScanningWorkerFilesystem(
        Service.getSharedCache(), RealFS, PPSkipMappings.get());
  if (Service.canReuseFileManager())
    Files = new FileManager(FileSystemOptions(), RealFS);
}

static llvm::Error runWithDiags(
    DiagnosticOptions *DiagOpts,
    llvm::function_ref<bool(DiagnosticConsumer &DC)> BodyShouldSucceed) {
  // Capture the emitted diagnostics and report them to the client
  // in the case of a failure.
  std::string DiagnosticOutput;
  llvm::raw_string_ostream DiagnosticsOS(DiagnosticOutput);
  TextDiagnosticPrinter DiagPrinter(DiagnosticsOS, DiagOpts);

  if (BodyShouldSucceed(DiagPrinter))
    return llvm::Error::success();
  return llvm::make_error<llvm::StringError>(DiagnosticsOS.str(),
                                             llvm::inconvertibleErrorCode());
}

llvm::Error DependencyScanningWorker::computeDependencies(
    const std::string &Input, StringRef WorkingDirectory,
    const CompilationDatabase &CDB, DependencyConsumer &Consumer) {
  RealFS->setCurrentWorkingDirectory(WorkingDirectory);
  return runWithDiags(DiagOpts.get(), [&](DiagnosticConsumer &DC) {
    /// Create the tool that uses the underlying file system to ensure that any
    /// file system requests that are made by the driver do not go through the
    /// dependency scanning filesystem.
    tooling::ClangTool Tool(CDB, Input, PCHContainerOps, RealFS, Files);
    Tool.clearArgumentsAdjusters();
    Tool.setRestoreWorkingDir(false);
    Tool.setPrintErrorMessage(false);
    Tool.setDiagnosticConsumer(&DC);
    DependencyScanningAction Action(WorkingDirectory, Consumer, DepFS,
                                    PPSkipMappings.get(), Format);
    return !Tool.run(&Action);
  });
}
