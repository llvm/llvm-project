//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_DEPENDENCYSCANNINGTOOL_H
#define LLVM_CLANG_TOOLING_DEPENDENCYSCANNINGTOOL_H

#include "clang/DependencyScanning/DependencyScannerImpl.h"
#include "clang/DependencyScanning/DependencyScanningService.h"
#include "clang/DependencyScanning/DependencyScanningUtils.h"
#include "clang/DependencyScanning/DependencyScanningWorker.h"
#include "clang/DependencyScanning/ModuleDepCollector.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/DenseSet.h"
#include <optional>
#include <string>
#include <vector>

namespace clang {
namespace tooling {

struct P1689Rule {
  std::string PrimaryOutput;
  std::optional<dependencies::P1689ModuleInfo> Provides;
  std::vector<dependencies::P1689ModuleInfo> Requires;
};

/// The high-level implementation of the dependency discovery tool that runs on
/// an individual worker thread.
class DependencyScanningTool {
public:
  /// Construct a dependency scanning tool.
  ///
  /// @param Service  The parent service. Must outlive the tool.
  /// @param FS The filesystem for the tool to use. Defaults to the physical FS.
  DependencyScanningTool(dependencies::DependencyScanningService &Service,
                         llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS =
                             llvm::vfs::createPhysicalFileSystem());

  /// Print out the dependency information into a string using the dependency
  /// file format that is specified in the options (-MD is the default) and
  /// return it.
  ///
  /// \returns A \c StringError with the diagnostic output if clang errors
  /// occurred, dependency file contents otherwise.
  std::optional<std::string>
  getDependencyFile(StringRef CWD, ArrayRef<std::string> CommandLine,
                    DiagnosticConsumer &DiagConsumer);

  /// Collect the module dependency in P1689 format for C++20 named modules.
  ///
  /// \param MakeformatOutput The output parameter for dependency information
  /// in make format if the command line requires to generate make-format
  /// dependency information by `-MD -MF <dep_file>`.
  ///
  /// \param MakeformatOutputPath The output parameter for the path to
  /// \param MakeformatOutput.
  ///
  /// \returns A \c StringError with the diagnostic output if clang errors
  /// occurred, P1689 dependency format rules otherwise.
  std::optional<P1689Rule>
  getP1689ModuleDependencyFile(const CompileCommand &Command, StringRef CWD,
                               std::string &MakeformatOutput,
                               std::string &MakeformatOutputPath,
                               DiagnosticConsumer &DiagConsumer);
  std::optional<P1689Rule>
  getP1689ModuleDependencyFile(const CompileCommand &Command, StringRef CWD,
                               DiagnosticConsumer &DiagConsumer) {
    std::string MakeformatOutput;
    std::string MakeformatOutputPath;

    return getP1689ModuleDependencyFile(Command, CWD, MakeformatOutput,
                                        MakeformatOutputPath, DiagConsumer);
  }

  /// Given a Clang driver command-line for a translation unit, gather the
  /// modular dependencies and return the information needed for explicit build.
  ///
  /// \param AlreadySeen This stores modules which have previously been
  ///                    reported. Use the same instance for all calls to this
  ///                    function for a single \c DependencyScanningTool in a
  ///                    single build. Use a different one for different tools,
  ///                    and clear it between builds.
  /// \param LookupModuleOutput This function is called to fill in
  ///                           "-fmodule-file=", "-o" and other output
  ///                           arguments for dependencies.
  /// \param TUBuffer Optional memory buffer for translation unit input. If
  ///                 TUBuffer is nullopt, the input should be included in the
  ///                 Commandline already.
  ///
  /// \returns a \c StringError with the diagnostic output if clang errors
  /// occurred, \c TranslationUnitDeps otherwise.
  std::optional<dependencies::TranslationUnitDeps>
  getTranslationUnitDependencies(
      ArrayRef<std::string> CommandLine, StringRef CWD,
      DiagnosticConsumer &DiagConsumer,
      const llvm::DenseSet<dependencies::ModuleID> &AlreadySeen,
      dependencies::LookupModuleOutputCallback LookupModuleOutput,
      std::optional<llvm::MemoryBufferRef> TUBuffer = std::nullopt);

  /// Given a compilation context specified via the Clang driver command-line,
  /// gather modular dependencies of module with the given name, and return the
  /// information needed for explicit build.
  /// TODO: this method should be removed as soon as Swift and our C-APIs adopt
  /// CompilerInstanceWithContext. We are keeping it here so that it is easier
  /// to coordinate with Swift and C-API changes.
  llvm::Expected<dependencies::TranslationUnitDeps> getModuleDependencies(
      StringRef ModuleName, ArrayRef<std::string> CommandLine, StringRef CWD,
      const llvm::DenseSet<dependencies::ModuleID> &AlreadySeen,
      dependencies::LookupModuleOutputCallback LookupModuleOutput);

  /// The following three methods provide a new interface to perform
  /// by name dependency scan. The new interface's intention is to improve
  /// dependency scanning performance when a sequence of name is looked up
  /// with the same current working directory and the command line.

  /// @brief Initializing the context and the compiler instance.
  ///        This method must be called before calling
  ///        computeDependenciesByNameWithContext.
  /// @param CWD The current working directory used during the scan.
  /// @param CommandLine The commandline used for the scan.
  /// @return Error if the initializaiton fails.
  llvm::Error initializeCompilerInstanceWithContextOrError(
      StringRef CWD, ArrayRef<std::string> CommandLine);

  /// @brief Computes the dependeny for the module named ModuleName.
  /// @param ModuleName The name of the module for which this method computes
  ///.                  dependencies.
  /// @param AlreadySeen This stores modules which have previously been
  ///                    reported. Use the same instance for all calls to this
  ///                    function for a single \c DependencyScanningTool in a
  ///                    single build. Note that this parameter is not part of
  ///                    the context because it can be shared across different
  ///                    worker threads and each worker thread may update it.
  /// @param LookupModuleOutput This function is called to fill in
  ///                           "-fmodule-file=", "-o" and other output
  ///                           arguments for dependencies.
  /// @return An instance of \c TranslationUnitDeps if the scan is successful.
  ///         Otherwise it returns an error.
  llvm::Expected<dependencies::TranslationUnitDeps>
  computeDependenciesByNameWithContextOrError(
      StringRef ModuleName,
      const llvm::DenseSet<dependencies::ModuleID> &AlreadySeen,
      dependencies::LookupModuleOutputCallback LookupModuleOutput);

  /// @brief This method finializes the compiler instance. It finalizes the
  ///        diagnostics and deletes the compiler instance. Call this method
  ///        once all names for a same commandline are scanned.
  /// @return Error if an error occured during finalization.
  llvm::Error finalizeCompilerInstanceWithContextOrError();

  llvm::vfs::FileSystem &getWorkerVFS() const { return Worker.getVFS(); }

private:
  dependencies::DependencyScanningWorker Worker;
  std::unique_ptr<dependencies::TextDiagnosticsPrinterWithOutput>
      DiagPrinterWithOS;
};

} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_DEPENDENCYSCANNINGTOOL_H
