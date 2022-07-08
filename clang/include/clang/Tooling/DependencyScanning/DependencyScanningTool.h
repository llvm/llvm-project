//===- DependencyScanningTool.h - clang-scan-deps service -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_DEPENDENCYSCANNING_DEPENDENCYSCANNINGTOOL_H
#define LLVM_CLANG_TOOLING_DEPENDENCYSCANNING_DEPENDENCYSCANNINGTOOL_H

#include "clang/Tooling/DependencyScanning/DependencyScanningService.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningWorker.h"
#include "clang/Tooling/DependencyScanning/ModuleDepCollector.h"
#include "clang/Tooling/JSONCompilationDatabase.h"
#include "llvm/ADT/StringSet.h"
#include <string>

namespace clang {
namespace tooling {
namespace dependencies {

/// The full dependencies and module graph for a specific input.
struct FullDependencies {
  /// The identifier of the C++20 module this translation unit exports.
  ///
  /// If the translation unit is not a module then \c ID.ModuleName is empty.
  ModuleID ID;

  /// A collection of absolute paths to files that this translation unit
  /// directly depends on, not including transitive dependencies.
  std::vector<std::string> FileDeps;

  /// A collection of prebuilt modules this translation unit directly depends
  /// on, not including transitive dependencies.
  std::vector<PrebuiltModuleDep> PrebuiltModuleDeps;

  /// A list of modules this translation unit directly depends on, not including
  /// transitive dependencies.
  ///
  /// This may include modules with a different context hash when it can be
  /// determined that the differences are benign for this compilation.
  std::vector<ModuleID> ClangModuleDeps;

  /// The original command line of the TU (excluding the compiler executable).
  std::vector<std::string> OriginalCommandLine;

  /// Get the full command line.
  ///
  /// \param LookupModuleOutput This function is called to fill in
  ///                           "-fmodule-file=", "-o" and other output
  ///                           arguments for dependencies.
  std::vector<std::string> getCommandLine(
      llvm::function_ref<std::string(const ModuleID &, ModuleOutputKind)>
          LookupOutput) const;

  /// Get the full command line, excluding -fmodule-file=" arguments.
  std::vector<std::string> getCommandLineWithoutModulePaths() const;
};

struct FullDependenciesResult {
  FullDependencies FullDeps;
  std::vector<ModuleDeps> DiscoveredModules;
};

/// The high-level implementation of the dependency discovery tool that runs on
/// an individual worker thread.
class DependencyScanningTool {
public:
  /// Construct a dependency scanning tool.
  DependencyScanningTool(DependencyScanningService &Service);

  /// Print out the dependency information into a string using the dependency
  /// file format that is specified in the options (-MD is the default) and
  /// return it. If \p ModuleName isn't empty, this function returns the
  /// dependency information of module \p ModuleName.
  ///
  /// \returns A \c StringError with the diagnostic output if clang errors
  /// occurred, dependency file contents otherwise.
  llvm::Expected<std::string>
  getDependencyFile(const std::vector<std::string> &CommandLine, StringRef CWD,
                    llvm::Optional<StringRef> ModuleName = None);

  /// Collect the full module dependency graph for the input, ignoring any
  /// modules which have already been seen. If \p ModuleName isn't empty, this
  /// function returns the full dependency information of module \p ModuleName.
  ///
  /// \param AlreadySeen This stores modules which have previously been
  ///                    reported. Use the same instance for all calls to this
  ///                    function for a single \c DependencyScanningTool in a
  ///                    single build. Use a different one for different tools,
  ///                    and clear it between builds.
  ///
  /// \returns a \c StringError with the diagnostic output if clang errors
  /// occurred, \c FullDependencies otherwise.
  llvm::Expected<FullDependenciesResult>
  getFullDependencies(const std::vector<std::string> &CommandLine,
                      StringRef CWD, const llvm::StringSet<> &AlreadySeen,
                      llvm::Optional<StringRef> ModuleName = None);

private:
  DependencyScanningWorker Worker;
};

} // end namespace dependencies
} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_DEPENDENCYSCANNING_DEPENDENCYSCANNINGTOOL_H
