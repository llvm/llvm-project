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
  DependencyScanningTool(dependencies::DependencyScanningService &Service)
      : Worker(Service) {}

  /// Print out the dependency information into a string using the dependency
  /// file format that is specified in the options (-MD is the default) and
  /// return it.
  ///
  /// \returns std::nullopt if errors occurred (reported to the DiagConsumer),
  /// dependency file contents otherwise.
  std::optional<std::string>
  getDependencyFile(ArrayRef<std::string> CommandLine, StringRef CWD,
                    dependencies::LookupModuleOutputCallback LookupModuleOutput,
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
  /// \returns std::nullopt if errors occurred (reported to the DiagConsumer),
  /// P1689 dependency format rules otherwise.
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
  /// \returns std::nullopt if errors occurred (reported to the DiagConsumer),
  /// translation unit dependencies otherwise.
  std::optional<dependencies::TranslationUnitDeps>
  getTranslationUnitDependencies(
      ArrayRef<std::string> CommandLine, StringRef CWD,
      DiagnosticConsumer &DiagConsumer,
      const llvm::DenseSet<dependencies::ModuleID> &AlreadySeen,
      dependencies::LookupModuleOutputCallback LookupModuleOutput,
      std::optional<llvm::MemoryBufferRef> TUBuffer = std::nullopt);

  /// By-name scanning given a Clang command-line. The scanning context is
  /// initialized once and reused for each name pulled from
  /// \p getNextName until it returns std::nullopt. Results flow to
  /// \p DepConsumer (with per-name status via finishQuery); diagnostics flow to
  /// \p DiagConsumer.
  ///
  /// \returns true if the scanning context initialized successfully and all
  /// scans succeeded. Query the DepConsumer for the state of each individual
  /// scan.
  bool getByNameDependencies(
      StringRef CWD, ArrayRef<std::string> CommandLine,
      DiagnosticConsumer &DiagConsumer,
      dependencies::DependencyActionController &Controller,
      llvm::function_ref<std::optional<std::string>()> getNextName,
      dependencies::DependencyConsumer &DepConsumer);

  /// Returns the worker tracing VFS, if it was requested via the service.
  llvm::vfs::TracingFileSystem *getWorkerTracingVFS() const {
    return Worker.getTracingVFS();
  }

private:
  dependencies::DependencyScanningWorker Worker;
};

/// Run the dependency scanning worker for the given driver or frontend
/// command-line, and report the discovered dependencies to the provided
/// consumer.
///
/// OverlayFS should be based on the Worker's dependency scanning file-system
/// and can be used to provide any input specified on the command-line as
/// in-memory file. If no overlay file-system is provided, the Worker's
/// dependency scanning file-system is used instead.
///
/// \returns false if any errors occurred (with diagnostics reported to
/// \c DiagConsumer), true otherwise.
bool computeDependencies(
    dependencies::DependencyScanningWorker &Worker, StringRef WorkingDirectory,
    ArrayRef<std::string> CommandLine,
    dependencies::DependencyConsumer &Consumer,
    dependencies::DependencyActionController &Controller,
    DiagnosticConsumer &DiagConsumer,
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> OverlayFS = nullptr);
} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_DEPENDENCYSCANNINGTOOL_H
