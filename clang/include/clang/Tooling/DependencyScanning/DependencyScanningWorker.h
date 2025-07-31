//===- DependencyScanningWorker.h - clang-scan-deps worker ===---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_DEPENDENCYSCANNING_DEPENDENCYSCANNINGWORKER_H
#define LLVM_CLANG_TOOLING_DEPENDENCYSCANNING_DEPENDENCYSCANNINGWORKER_H

#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LLVM.h"
#include "clang/Frontend/PCHContainerOperations.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningService.h"
#include "clang/Tooling/DependencyScanning/ModuleDepCollector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBufferRef.h"
#include <optional>
#include <string>

namespace clang {

class DependencyOutputOptions;

namespace tooling {
namespace dependencies {

class DependencyScanningWorkerFilesystem;

/// A command-line tool invocation that is part of building a TU.
///
/// \see TranslationUnitDeps::Commands.
struct Command {
  std::string Executable;
  std::vector<std::string> Arguments;
};

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

/// Dependency scanner callbacks that are used during scanning to influence the
/// behaviour of the scan - for example, to customize the scanned invocations.
class DependencyActionController {
public:
  virtual ~DependencyActionController();

  virtual std::string lookupModuleOutput(const ModuleDeps &MD,
                                         ModuleOutputKind Kind) = 0;
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
  /// @param FS The filesystem for the worker to use.
  DependencyScanningWorker(DependencyScanningService &Service,
                           llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS);

  /// Run the dependency scanning tool for a given clang driver command-line,
  /// and report the discovered dependencies to the provided consumer. If
  /// TUBuffer is not nullopt, it is used as TU input for the dependency
  /// scanning. Otherwise, the input should be included as part of the
  /// command-line.
  ///
  /// \returns false if clang errors occurred (with diagnostics reported to
  /// \c DiagConsumer), true otherwise.
  bool computeDependencies(
      StringRef WorkingDirectory, const std::vector<std::string> &CommandLine,
      DependencyConsumer &DepConsumer, DependencyActionController &Controller,
      DiagnosticConsumer &DiagConsumer,
      std::optional<llvm::MemoryBufferRef> TUBuffer = std::nullopt);

  /// Run the dependency scanning tool for a given clang driver command-line
  /// for a specific module.
  ///
  /// \returns false if clang errors occurred (with diagnostics reported to
  /// \c DiagConsumer), true otherwise.
  bool computeDependencies(StringRef WorkingDirectory,
                           const std::vector<std::string> &CommandLine,
                           DependencyConsumer &DepConsumer,
                           DependencyActionController &Controller,
                           DiagnosticConsumer &DiagConsumer,
                           StringRef ModuleName);

  /// Run the dependency scanning tool for a given clang driver command-line
  /// for a specific translation unit via file system or memory buffer.
  ///
  /// \returns A \c StringError with the diagnostic output if clang errors
  /// occurred, success otherwise.
  llvm::Error computeDependencies(
      StringRef WorkingDirectory, const std::vector<std::string> &CommandLine,
      DependencyConsumer &Consumer, DependencyActionController &Controller,
      std::optional<llvm::MemoryBufferRef> TUBuffer = std::nullopt);

  /// Run the dependency scanning tool for a given clang driver command-line
  /// for a specific module.
  ///
  /// \returns A \c StringError with the diagnostic output if clang errors
  /// occurred, success otherwise.
  llvm::Error computeDependencies(StringRef WorkingDirectory,
                                  const std::vector<std::string> &CommandLine,
                                  DependencyConsumer &Consumer,
                                  DependencyActionController &Controller,
                                  StringRef ModuleName);

  llvm::vfs::FileSystem &getVFS() const { return *BaseFS; }

private:
  /// The parent dependency scanning service.
  DependencyScanningService &Service;
  std::shared_ptr<PCHContainerOperations> PCHContainerOps;
  /// The file system to be used during the scan.
  /// This is either \c FS passed in the constructor (when performing canonical
  /// preprocessing), or \c DepFS (when performing dependency directives scan).
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> BaseFS;
  /// When performing dependency directives scan, this is the caching (and
  /// dependency-directives-extracting) filesystem overlaid on top of \c FS
  /// (passed in the constructor).
  llvm::IntrusiveRefCntPtr<DependencyScanningWorkerFilesystem> DepFS;

  /// Private helper functions.
  bool scanDependencies(StringRef WorkingDirectory,
                        const std::vector<std::string> &CommandLine,
                        DependencyConsumer &Consumer,
                        DependencyActionController &Controller,
                        DiagnosticConsumer &DC,
                        llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS,
                        std::optional<StringRef> ModuleName);
};

} // end namespace dependencies
} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_DEPENDENCYSCANNING_DEPENDENCYSCANNINGWORKER_H
