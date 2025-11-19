//===- DependencyScannerImpl.h - Implements dependency scanning *- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_DEPENDENCYSCANNING_DEPENDENCYSCANNER_H
#define LLVM_CLANG_TOOLING_DEPENDENCYSCANNING_DEPENDENCYSCANNER_H

#include "clang/Driver/Compilation.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Serialization/ObjectFilePCHContainerReader.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningFilesystem.h"
#include "clang/Tooling/DependencyScanning/ModuleDepCollector.h"

namespace clang {
class DiagnosticConsumer;

namespace tooling {
namespace dependencies {
class DependencyScanningService;
class DependencyScanningWorker;

class DependencyConsumer;
class DependencyActionController;
class DependencyScanningWorkerFilesystem;

class DependencyScanningAction {
public:
  DependencyScanningAction(
      DependencyScanningService &Service, StringRef WorkingDirectory,
      DependencyConsumer &Consumer, DependencyActionController &Controller,
      llvm::IntrusiveRefCntPtr<DependencyScanningWorkerFilesystem> DepFS,
      llvm::IntrusiveRefCntPtr<DependencyScanningCASFilesystem> DepCASFS,
      llvm::IntrusiveRefCntPtr<llvm::cas::CachingOnDiskFileSystem> CacheFS,
      bool EmitDependencyFile, bool DiagGenerationAsCompilation,
      const CASOptions &CASOpts,
      std::optional<StringRef> ModuleName = std::nullopt,
      raw_ostream *VerboseOS = nullptr)
      : Service(Service), WorkingDirectory(WorkingDirectory),
        Consumer(Consumer), Controller(Controller), DepFS(std::move(DepFS)),
        DepCASFS(std::move(DepCASFS)), CacheFS(std::move(CacheFS)),
        CASOpts(CASOpts), EmitDependencyFile(EmitDependencyFile),
        DiagGenerationAsCompilation(DiagGenerationAsCompilation),
        VerboseOS(VerboseOS) {}
  bool runInvocation(std::shared_ptr<CompilerInvocation> Invocation,
                     IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS,
                     std::shared_ptr<PCHContainerOperations> PCHContainerOps,
                     DiagnosticConsumer *DiagConsumer);

  bool hasScanned() const { return Scanned; }
  bool hasDiagConsumerFinished() const { return DiagConsumerFinished; }

  /// Take the cc1 arguments corresponding to the most recent invocation used
  /// with this action. Any modifications implied by the discovered dependencies
  /// will have already been applied.
  std::vector<std::string> takeLastCC1Arguments();

  std::optional<std::string> takeLastCC1CacheKey();

  IntrusiveRefCntPtr<llvm::vfs::FileSystem> getDepScanFS();

private:
  DependencyScanningService &Service;
  StringRef WorkingDirectory;
  DependencyConsumer &Consumer;
  DependencyActionController &Controller;
  llvm::IntrusiveRefCntPtr<DependencyScanningWorkerFilesystem> DepFS;
  llvm::IntrusiveRefCntPtr<DependencyScanningCASFilesystem> DepCASFS;
  llvm::IntrusiveRefCntPtr<llvm::cas::CachingOnDiskFileSystem> CacheFS;
  const CASOptions &CASOpts;
  bool EmitDependencyFile = false;
  bool DiagGenerationAsCompilation;
  std::optional<StringRef> ModuleName;
  std::optional<CompilerInstance> ScanInstanceStorage;
  std::shared_ptr<ModuleDepCollector> MDC;
  std::vector<std::string> LastCC1Arguments;
  std::optional<std::string> LastCC1CacheKey;
  bool Scanned = false;
  bool DiagConsumerFinished = false;
  raw_ostream *VerboseOS;
};

// Helper functions and data types.
std::unique_ptr<DiagnosticOptions>
createDiagOptions(ArrayRef<std::string> CommandLine);

struct DignosticsEngineWithDiagOpts {
  // We need to bound the lifetime of the DiagOpts used to create the
  // DiganosticsEngine with the DiagnosticsEngine itself.
  std::unique_ptr<DiagnosticOptions> DiagOpts;
  IntrusiveRefCntPtr<DiagnosticsEngine> DiagEngine;

  DignosticsEngineWithDiagOpts(ArrayRef<std::string> CommandLine,
                               IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS,
                               DiagnosticConsumer &DC);
};

struct TextDiagnosticsPrinterWithOutput {
  // We need to bound the lifetime of the data that supports the DiagPrinter
  // with it together so they have the same lifetime.
  std::string DiagnosticOutput;
  llvm::raw_string_ostream DiagnosticsOS;
  std::unique_ptr<DiagnosticOptions> DiagOpts;
  TextDiagnosticPrinter DiagPrinter;

  TextDiagnosticsPrinterWithOutput(ArrayRef<std::string> CommandLine)
      : DiagnosticsOS(DiagnosticOutput),
        DiagOpts(createDiagOptions(CommandLine)),
        DiagPrinter(DiagnosticsOS, *DiagOpts) {}
};

std::pair<std::unique_ptr<driver::Driver>, std::unique_ptr<driver::Compilation>>
buildCompilation(ArrayRef<std::string> ArgStrs, DiagnosticsEngine &Diags,
                 IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS,
                 llvm::BumpPtrAllocator &Alloc);

std::unique_ptr<CompilerInvocation>
createCompilerInvocation(ArrayRef<std::string> CommandLine,
                         DiagnosticsEngine &Diags);

std::pair<IntrusiveRefCntPtr<llvm::vfs::FileSystem>, std::vector<std::string>>
initVFSForTUBuferScanning(
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> BaseFS,
    ArrayRef<std::string> CommandLine, StringRef WorkingDirectory,
    llvm::MemoryBufferRef TUBuffer, std::shared_ptr<cas::ObjectStore> CAS,
    IntrusiveRefCntPtr<DependencyScanningCASFilesystem> DepCASFS);

std::pair<IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem>,
          std::vector<std::string>>
initVFSForByNameScanning(
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> BaseFS,
    ArrayRef<std::string> CommandLine, StringRef WorkingDirectory,
    StringRef ModuleName, std::shared_ptr<cas::ObjectStore> CAS,
    IntrusiveRefCntPtr<DependencyScanningCASFilesystem> DepCASFS);

bool initializeScanCompilerInstance(
    CompilerInstance &ScanInstance,
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS,
    DiagnosticConsumer *DiagConsumer, DependencyScanningService &Service,
    IntrusiveRefCntPtr<DependencyScanningWorkerFilesystem> DepFS,
    bool DiagGenerationAsCompilation, raw_ostream *VerboseOS,
    llvm::IntrusiveRefCntPtr<DependencyScanningCASFilesystem> DepCASFS);

SmallVector<StringRef>
getInitialStableDirs(const CompilerInstance &ScanInstance);

std::optional<PrebuiltModulesAttrsMap>
computePrebuiltModulesASTMap(CompilerInstance &ScanInstance,
                             SmallVector<StringRef> &StableDirs);

std::unique_ptr<DependencyOutputOptions>
takeAndUpdateDependencyOutputOptionsFrom(CompilerInstance &ScanInstance,
                                         bool ForceIncludeSystemHeaders);

/// Create the dependency collector that will collect the produced
/// dependencies. May return the created ModuleDepCollector depending
/// on the scanning format.
std::shared_ptr<ModuleDepCollector> initializeScanInstanceDependencyCollector(
    CompilerInstance &ScanInstance,
    std::unique_ptr<DependencyOutputOptions> DepOutputOpts,
    StringRef WorkingDirectory, DependencyConsumer &Consumer,
    DependencyScanningService &Service, CompilerInvocation &Inv,
    DependencyActionController &Controller,
    PrebuiltModulesAttrsMap PrebuiltModulesASTMap,
    llvm::SmallVector<StringRef> &StableDirs, bool EmitDependencyFile);

class CompilerInstanceWithContext {
  // Context
  DependencyScanningWorker &Worker;
  llvm::StringRef CWD;
  std::vector<std::string> CommandLine;

  // Context - file systems
  llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> OverlayFS;

  // Context - Diagnostics engine.
  std::unique_ptr<TextDiagnosticsPrinterWithOutput> DiagPrinterWithOS;
  // DiagConsumer may points to DiagPrinterWithOS->DiagPrinter, or a custom
  // DiagnosticConsumer passed in from initialize.
  DiagnosticConsumer *DiagConsumer = nullptr;
  std::unique_ptr<DignosticsEngineWithDiagOpts> DiagEngineWithCmdAndOpts;

  // Context - compiler invocation
  // Compilation's command's arguments may be owned by Alloc when expanded from
  // response files, so we need to keep Alloc alive in the context.
  llvm::BumpPtrAllocator Alloc;
  std::unique_ptr<clang::driver::Driver> Driver;
  std::unique_ptr<clang::driver::Compilation> Compilation;
  std::unique_ptr<CompilerInvocation> OriginalInvocation;

  // Context - output options
  std::unique_ptr<DependencyOutputOptions> OutputOpts;

  // Context - stable directory handling
  llvm::SmallVector<StringRef> StableDirs;
  PrebuiltModulesAttrsMap PrebuiltModuleASTMap;

  // Compiler Instance
  std::unique_ptr<CompilerInstance> CIPtr;

  // Source location offset.
  int32_t SrcLocOffset = 0;

public:
  CompilerInstanceWithContext(DependencyScanningWorker &Worker, StringRef CWD,
                              const std::vector<std::string> &CMD)
      : Worker(Worker), CWD(CWD), CommandLine(CMD) {};

  // The three methods below returns false when they fail, with the detail
  // accumulated in DiagConsumer.
  bool initialize(DiagnosticConsumer *DC);
  bool computeDependencies(StringRef ModuleName, DependencyConsumer &Consumer,
                           DependencyActionController &Controller);
  bool finalize();

  // The method below turns the return status from the above methods
  // into an llvm::Error using a default DiagnosticConsumer.
  llvm::Error handleReturnStatus(bool Success);
};
} // namespace dependencies
} // namespace tooling
} // namespace clang

#endif
