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
#include <string>

namespace clang {

class DependencyOutputOptions;

namespace tooling {
namespace dependencies {

class DependencyScanningWorkerFilesystem;

/// A command-line tool invocation that is part of building a TU.
///
/// \see FullDependencies::Commands.
struct Command {
  std::string Executable;
  std::vector<std::string> Arguments;
};

using RemapPathCallback =
    llvm::function_ref<StringRef(const llvm::vfs::CachedDirectoryEntry &)>;

class DependencyConsumer {
public:
  virtual ~DependencyConsumer() {}

  virtual void finalize(CompilerInstance &CI) {}

  virtual void handleBuildCommand(Command Cmd) = 0;

  virtual void
  handleDependencyOutputOpts(const DependencyOutputOptions &Opts) = 0;

  virtual void handleFileDependency(StringRef Filename) = 0;

  virtual void handlePrebuiltModuleDependency(PrebuiltModuleDep PMD) = 0;

  virtual void handleModuleDependency(ModuleDeps MD) = 0;

  virtual void handleContextHash(std::string Hash) = 0;

  virtual void handleCASFileSystemRootID(cas::CASID ID) = 0;

  virtual std::string lookupModuleOutput(const ModuleID &ID,
                                         ModuleOutputKind Kind) = 0;
};

// FIXME: This may need to merge with \p DependencyConsumer in order to support
// clang modules for the include-tree.
class PPIncludeActionsConsumer : public DependencyConsumer {
public:
  virtual void enteredInclude(Preprocessor &PP, FileID FID) = 0;

  virtual void exitedInclude(Preprocessor &PP, FileID IncludedBy,
                             FileID Include, SourceLocation ExitLoc) = 0;

  virtual void handleHasIncludeCheck(Preprocessor &PP, bool Result) = 0;

protected:
  void handleBuildCommand(Command) override {}
  void handleDependencyOutputOpts(const DependencyOutputOptions &Opts) override {
    llvm::report_fatal_error("unexpected callback for include-tree");
  }
  void handleFileDependency(StringRef Filename) override {
    llvm::report_fatal_error("unexpected callback for include-tree");
  }
  void handlePrebuiltModuleDependency(PrebuiltModuleDep PMD) override {
    llvm::report_fatal_error("unexpected callback for include-tree");
  }
  void handleModuleDependency(ModuleDeps MD) override {
    llvm::report_fatal_error("unexpected callback for include-tree");
  }
  void handleContextHash(std::string Hash) override {
    llvm::report_fatal_error("unexpected callback for include-tree");
  }
  void handleCASFileSystemRootID(cas::CASID ID) override {
    llvm::report_fatal_error("unexpected callback for include-tree");
  }
  std::string lookupModuleOutput(const ModuleID &, ModuleOutputKind) override {
    llvm::report_fatal_error("unexpected callback for include-tree");
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
  /// \returns A \c StringError with the diagnostic output if clang errors
  /// occurred, success otherwise.
  llvm::Error computeDependencies(StringRef WorkingDirectory,
                                  const std::vector<std::string> &CommandLine,
                                  DependencyConsumer &Consumer,
                                  llvm::Optional<StringRef> ModuleName = None);

  ScanningOutputFormat getFormat() const { return Format; }

  /// Scan from a compiler invocation.
  /// If \p DiagGenerationAsCompilation is true it will generate error
  /// diagnostics same way as the normal compilation, with "N errors generated"
  /// message and the serialized diagnostics file emitted if the
  /// \p DiagOpts.DiagnosticSerializationFile setting is set for the invocation.
  void computeDependenciesFromCompilerInvocation(
      std::shared_ptr<CompilerInvocation> Invocation,
      StringRef WorkingDirectory, DependencyConsumer &Consumer,
      RemapPathCallback RemapPath, DiagnosticConsumer &DiagsConsumer,
      raw_ostream *VerboseOS, bool DiagGenerationAsCompilation);

  ScanningOutputFormat getScanningFormat() const { return Format; }

  llvm::vfs::FileSystem &getRealFS() { return *RealFS; }
  llvm::cas::CachingOnDiskFileSystem &getCASFS() { return *CacheFS; }
  bool useCAS() const { return UseCAS; }
  const CASOptions &getCASOpts() const { return CASOpts; }

  /// If \p DependencyScanningService enabled sharing of \p FileManager this
  /// will return the same instance, otherwise it will create a new one for
  /// each invocation.
  llvm::IntrusiveRefCntPtr<FileManager> getOrCreateFileManager() const;

  bool shouldEagerLoadModules() const { return EagerLoadModules; }

private:
  std::shared_ptr<PCHContainerOperations> PCHContainerOps;

  /// The physical filesystem overlaid by `InMemoryFS`.
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> RealFS;
  /// The in-memory filesystem laid on top the physical filesystem in `RealFS`.
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFS;
  /// The file system that is used by each worker when scanning for
  /// dependencies. This filesystem persists across multiple compiler
  /// invocations.
  llvm::IntrusiveRefCntPtr<DependencyScanningWorkerFilesystem> DepFS;
  /// The file manager that is reused across multiple invocations by this
  /// worker. If null, the file manager will not be reused.
  llvm::IntrusiveRefCntPtr<FileManager> Files;
  ScanningOutputFormat Format;
  /// Whether to optimize the modules' command-line arguments.
  bool OptimizeArgs;

  /// The caching file system.
  llvm::IntrusiveRefCntPtr<llvm::cas::CachingOnDiskFileSystem> CacheFS;
  /// The CAS Dependency Filesytem. This is not set at the sametime as DepFS;
  llvm::IntrusiveRefCntPtr<DependencyScanningCASFilesystem> DepCASFS;
  CASOptions CASOpts;
  bool UseCAS;

  /// Whether to set up command-lines to load PCM files eagerly.
  bool EagerLoadModules;
};

} // end namespace dependencies
} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_DEPENDENCYSCANNING_DEPENDENCYSCANNINGWORKER_H
