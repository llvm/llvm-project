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
#include "llvm/CAS/CachingOnDiskFileSystem.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include <optional>
#include <string>

namespace clang {

class DependencyOutputOptions;

namespace tooling {
namespace dependencies {

using CachingOnDiskFileSystemPtr =
    llvm::IntrusiveRefCntPtr<llvm::cas::CachingOnDiskFileSystem>;

class DependencyScanningWorkerFilesystem;

/// A command-line tool invocation that is part of building a TU.
///
/// \see TranslationUnitDeps::Commands.
struct Command {
  std::string Executable;
  std::vector<std::string> Arguments;

  /// The \c ActionCache key for this translation unit, if any.
  std::optional<std::string> TUCacheKey;
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

  virtual void handleContextHash(std::string Hash) = 0;

  virtual void handleCASFileSystemRootID(std::string ID) {}

  virtual void handleIncludeTreeID(std::string ID) {}
};

/// Dependency scanner callbacks that are used during scanning to influence the
/// behaviour of the scan - for example, to customize the scanned invocations.
class DependencyActionController {
public:
  virtual ~DependencyActionController();

  virtual std::string lookupModuleOutput(const ModuleID &ID,
                                         ModuleOutputKind Kind) = 0;

  virtual llvm::Error initialize(CompilerInstance &ScanInstance,
                                 CompilerInvocation &NewInvocation) {
    return llvm::Error::success();
  }

  virtual llvm::Error finalize(CompilerInstance &ScanInstance,
                               CompilerInvocation &NewInvocation) {
    return llvm::Error::success();
  }

  virtual llvm::Error
  initializeModuleBuild(CompilerInstance &ModuleScanInstance) {
    return llvm::Error::success();
  }

  virtual llvm::Error
  finalizeModuleBuild(CompilerInstance &ModuleScanInstance) {
    return llvm::Error::success();
  }

  virtual llvm::Error finalizeModuleInvocation(CowCompilerInvocation &CI,
                                               const ModuleDeps &MD) {
    return llvm::Error::success();
  }
};

/// An individual dependency scanning worker that is able to run on its own
/// thread.
///
/// The worker computes the dependencies for the input files by preprocessing
/// sources either using a fast mode where the source files are minimized, or
/// using the regular processing run.
class DependencyScanningWorker {
public:
  DependencyScanningWorker(DependencyScanningService &Service,
                           llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS);

  /// Run the dependency scanning tool for a given clang driver command-line,
  /// and report the discovered dependencies to the provided consumer. If \p
  /// ModuleName isn't empty, this function reports the dependencies of module
  /// \p ModuleName.
  ///
  /// \returns false if clang errors occurred (with diagnostics reported to
  /// \c DiagConsumer), true otherwise.
  bool computeDependencies(StringRef WorkingDirectory,
                           const std::vector<std::string> &CommandLine,
                           DependencyConsumer &DepConsumer,
                           DependencyActionController &Controller,
                           DiagnosticConsumer &DiagConsumer,
                           std::optional<StringRef> ModuleName = std::nullopt);
  /// \returns A \c StringError with the diagnostic output if clang errors
  /// occurred, success otherwise.
  llvm::Error computeDependencies(
      StringRef WorkingDirectory, const std::vector<std::string> &CommandLine,
      DependencyConsumer &Consumer, DependencyActionController &Controller,
      std::optional<StringRef> ModuleName = std::nullopt);

  /// Scan from a compiler invocation.
  /// If \p DiagGenerationAsCompilation is true it will generate error
  /// diagnostics same way as the normal compilation, with "N errors generated"
  /// message and the serialized diagnostics file emitted if the
  /// \p DiagOpts.DiagnosticSerializationFile setting is set for the invocation.
  void computeDependenciesFromCompilerInvocation(
      std::shared_ptr<CompilerInvocation> Invocation,
      StringRef WorkingDirectory, DependencyConsumer &Consumer,
      DependencyActionController &Controller, DiagnosticConsumer &DiagsConsumer,
      raw_ostream *VerboseOS, bool DiagGenerationAsCompilation);

  ScanningOutputFormat getScanningFormat() const { return Format; }

  CachingOnDiskFileSystemPtr getCASFS() { return CacheFS; }
  const CASOptions &getCASOpts() const { return CASOpts; }
  std::shared_ptr<cas::ObjectStore> getCAS() const { return CAS; }

  /// If \p DependencyScanningService enabled sharing of \p FileManager this
  /// will return the same instance, otherwise it will create a new one for
  /// each invocation.
  llvm::IntrusiveRefCntPtr<FileManager> getOrCreateFileManager() const;

  bool shouldEagerLoadModules() const { return EagerLoadModules; }

private:
  std::shared_ptr<PCHContainerOperations> PCHContainerOps;
  /// The file system to be used during the scan.
  /// This is either \c FS passed in the constructor (when performing canonical
  /// preprocessing), or \c DepFS (when performing dependency directives scan).
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> BaseFS;
  /// When performing dependency directives scan, this is the caching (and
  /// dependency-directives-extracting) filesystem overlaid on top of \c FS
  /// (passed in the constructor).
  llvm::IntrusiveRefCntPtr<DependencyScanningWorkerFilesystem> DepFS;
  ScanningOutputFormat Format;
  /// Whether to optimize the modules' command-line arguments.
  bool OptimizeArgs;
  /// Whether to set up command-lines to load PCM files eagerly.
  bool EagerLoadModules;

  /// The caching file system.
  CachingOnDiskFileSystemPtr CacheFS;
  /// The CAS Dependency Filesytem. This is not set at the sametime as DepFS;
  llvm::IntrusiveRefCntPtr<DependencyScanningCASFilesystem> DepCASFS;
  CASOptions CASOpts;
  std::shared_ptr<cas::ObjectStore> CAS;
};

} // end namespace dependencies
} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_DEPENDENCYSCANNING_DEPENDENCYSCANNINGWORKER_H
