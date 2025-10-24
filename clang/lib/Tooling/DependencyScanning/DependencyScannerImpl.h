//===- DependencyScanner.h - Performs module dependency scanning *- C++ -*-===//
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
class DependencyConsumer;
class DependencyActionController;
class DependencyScanningWorkerFilesystem;

class DependencyScanningAction {
public:
  DependencyScanningAction(
      DependencyScanningService &Service, StringRef WorkingDirectory,
      DependencyConsumer &Consumer, DependencyActionController &Controller,
      IntrusiveRefCntPtr<DependencyScanningWorkerFilesystem> DepFS,
      std::optional<StringRef> ModuleName = std::nullopt)
      : Service(Service), WorkingDirectory(WorkingDirectory),
        Consumer(Consumer), Controller(Controller), DepFS(std::move(DepFS)),
        ModuleName(ModuleName) {}
  bool runInvocation(std::unique_ptr<CompilerInvocation> Invocation,
                     IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS,
                     std::shared_ptr<PCHContainerOperations> PCHContainerOps,
                     DiagnosticConsumer *DiagConsumer);

  bool hasScanned() const { return Scanned; }
  bool hasDiagConsumerFinished() const { return DiagConsumerFinished; }

  /// Take the cc1 arguments corresponding to the most recent invocation used
  /// with this action. Any modifications implied by the discovered dependencies
  /// will have already been applied.
  std::vector<std::string> takeLastCC1Arguments() {
    std::vector<std::string> Result;
    std::swap(Result, LastCC1Arguments); // Reset LastCC1Arguments to empty.
    return Result;
  }

private:
  void setLastCC1Arguments(CompilerInvocation &&CI) {
    if (MDC)
      MDC->applyDiscoveredDependencies(CI);
    LastCC1Arguments = CI.getCC1CommandLine();
  }

  DependencyScanningService &Service;
  StringRef WorkingDirectory;
  DependencyConsumer &Consumer;
  DependencyActionController &Controller;
  IntrusiveRefCntPtr<DependencyScanningWorkerFilesystem> DepFS;
  std::optional<StringRef> ModuleName;
  std::optional<CompilerInstance> ScanInstanceStorage;
  std::shared_ptr<ModuleDepCollector> MDC;
  std::vector<std::string> LastCC1Arguments;
  bool Scanned = false;
  bool DiagConsumerFinished = false;
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
initVFSForTUBuferScanning(IntrusiveRefCntPtr<llvm::vfs::FileSystem> BaseFS,
                          ArrayRef<std::string> CommandLine,
                          StringRef WorkingDirectory,
                          llvm::MemoryBufferRef TUBuffer);

std::pair<IntrusiveRefCntPtr<llvm::vfs::FileSystem>, std::vector<std::string>>
initVFSForByNameScanning(IntrusiveRefCntPtr<llvm::vfs::FileSystem> BaseFS,
                         ArrayRef<std::string> CommandLine,
                         StringRef WorkingDirectory, StringRef ModuleName);

bool initializeScanCompilerInstance(
    CompilerInstance &ScanInstance,
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS,
    DiagnosticConsumer *DiagConsumer, DependencyScanningService &Service,
    IntrusiveRefCntPtr<DependencyScanningWorkerFilesystem> DepFS);

SmallVector<StringRef>
getInitialStableDirs(const CompilerInstance &ScanInstance);

std::optional<PrebuiltModulesAttrsMap>
computePrebuiltModulesASTMap(CompilerInstance &ScanInstance,
                             SmallVector<StringRef> &StableDirs);

std::unique_ptr<DependencyOutputOptions>
takeDependencyOutputOptionsFrom(CompilerInstance &ScanInstance);

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
    llvm::SmallVector<StringRef> &StableDirs);
} // namespace dependencies
} // namespace tooling
} // namespace clang

#endif
