//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DEPENDENCYSCANNING_DEPENDENCYSCANNINGWORKER_H
#define LLVM_CLANG_DEPENDENCYSCANNING_DEPENDENCYSCANNINGWORKER_H

#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LLVM.h"
#include "clang/DependencyScanning/DependencyScannerImpl.h"
#include "clang/DependencyScanning/DependencyScanningService.h"
#include "clang/DependencyScanning/ModuleDepCollector.h"
#include "clang/Frontend/PCHContainerOperations.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <optional>
#include <string>

namespace clang {

class DependencyOutputOptions;

namespace tooling {
class CompilerInstanceWithContext;
}

namespace dependencies {

class DependencyScanningWorkerFilesystem;

class DependencyConsumer {
public:
  virtual ~DependencyConsumer() {}

  virtual void handleProvidedAndRequiredStdCXXModules(
      std::optional<P1689ModuleInfo> Provided,
      std::vector<P1689ModuleInfo> Requires) {}

  virtual void handleBuildCommand(Command Cmd) {}

  virtual void
  handleDependencyOutputOpts(const DependencyOutputOptions &Opts) = 0;

  virtual void handleFileDependency(StringRef Filename) = 0;

  virtual void handlePrebuiltModuleDependency(PrebuiltModuleDep PMD) = 0;

  virtual void handleModuleDependency(ModuleDeps MD) = 0;

  virtual void handleDirectModuleDependency(ModuleID MD) = 0;

  virtual void handleVisibleModule(std::string ModuleName) = 0;

  virtual void handleContextHash(std::string Hash) = 0;
};

/// An individual dependency scanning worker that is able to run on its own
/// thread.
///
/// The worker computes the dependencies for the input files by preprocessing
/// sources either using a fast mode where the source files are minimized, or
/// using the regular processing run.
class DependencyScanningWorker {
public:
  /// Construct a dependency scanning worker.
  ///
  /// @param Service The parent service. Must outlive the worker.
  DependencyScanningWorker(DependencyScanningService &Service);

  ~DependencyScanningWorker();

  /// Run the dependency scanning tool for all given frontend command-lines,
  /// and report the discovered dependencies to the provided consumer.
  ///
  /// OverlayFS should be based on the Worker's dependency scanning file-system
  /// and can be used to provide any input specified on the command-line as
  /// in-memory file. If no overlay file-system is provided, the Worker's
  /// dependency scanning file-system is used instead.
  ///
  /// \returns false if any errors occurred (with diagnostics reported to
  /// \c DiagConsumer), true otherwise.
  bool computeDependencies(
      StringRef WorkingDirectory, ArrayRef<ArrayRef<std::string>> CommandLines,
      DependencyConsumer &DepConsumer, DependencyActionController &Controller,
      DiagnosticConsumer &DiagConsumer,
      IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> OverlayFS = nullptr);

  llvm::vfs::FileSystem &getVFS() const { return *DepFS; }

private:
  /// The parent dependency scanning service.
  DependencyScanningService &Service;
  std::shared_ptr<PCHContainerOperations> PCHContainerOps;
  /// This is the caching (and optionally dependency-directives-providing) VFS
  /// overlaid on top of the base VFS passed in the constructor.
  IntrusiveRefCntPtr<DependencyScanningWorkerFilesystem> DepFS;

  friend tooling::CompilerInstanceWithContext;
};
} // end namespace dependencies
} // end namespace clang

#endif // LLVM_CLANG_DEPENDENCYSCANNING_DEPENDENCYSCANNINGWORKER_H
