//===-------------- ModuleDependencyScanner.h --------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_MODULEDEPENDENCYSCANNER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_MODULEDEPENDENCYSCANNER_H

#include "GlobalCompilationDatabase.h"
#include "support/Path.h"
#include "support/ThreadsafeFS.h"

#include "clang/Tooling/DependencyScanning/DependencyScanningService.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningTool.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"

namespace clang {
namespace clangd {

/// A scanner to query the dependency information for C++20 Modules.
///
/// The scanner can scan a single file with `scan(PathRef)` member function
/// or scan the whole project with `globalScan(PathRef)` member function. See
/// the comments of `globalScan` to see the details.
///
/// ModuleDependencyScanner should only be used via ScanningAllProjectModules.
///
/// The ModuleDependencyScanner can get the directly required module name for a
/// specific source file. Also the ModuleDependencyScanner can get the source
/// file declaring a specific module name.
///
/// IMPORTANT NOTE: we assume that every module unit is only declared once in a
/// source file in the project. But the assumption is not strictly true even
/// besides the invalid projects. The language specification requires that every
/// module unit should be unique in a valid program. But a project can contain
/// multiple programs. Then it is valid that we can have multiple source files
/// declaring the same module in a project as long as these source files don't
/// interere with each other.`
class ModuleDependencyScanner {
public:
  ModuleDependencyScanner(const GlobalCompilationDatabase &CDB,
                          const ThreadsafeFS &TFS)
      : CDB(CDB), TFS(TFS),
        Service(tooling::dependencies::ScanningMode::CanonicalPreprocessing,
                tooling::dependencies::ScanningOutputFormat::P1689) {}

  // The scanned modules dependency information for a specific source file.
  struct ModuleDependencyInfo {
    // The name of the module if the file is a module unit.
    std::optional<std::string> ModuleName;
    // A list of names for the modules that the file directly depends.
    std::vector<std::string> RequiredModules;
  };

  /// Scanning the single file specified by \param FilePath.
  /// NOTE: This is only used by unittests for external uses.
  std::optional<ModuleDependencyInfo> scan(PathRef FilePath);

  /// Scanning every source file in the current project to get the
  /// <module-name> to <module-unit-source> map.
  /// It looks unefficiency to scan the whole project especially for
  /// every version of every file!
  /// TODO: We should find an efficient method to get the <module-name>
  /// to <module-unit-source> map. We can make it either by providing
  /// a global module dependency scanner to monitor every file. Or we
  /// can simply require the build systems (or even if the end users)
  /// to provide the map.
  void globalScan(const std::vector<std::string> &AllFiles);
  bool isGlobalScanned() const { return GlobalScanned; }

  /// Get the source file from the module name. Note that the language
  /// guarantees all the module names are unique in a valid program.
  /// This function should only be called after globalScan.
  ///
  /// FIXME: We should handle the case that there are multiple source files
  /// declaring the same module.
  PathRef getSourceForModuleName(StringRef ModuleName) const;

  /// Return the direct required modules. Indirect required modules are not
  /// included.
  std::vector<std::string> getRequiredModules(PathRef File);

private:
  const GlobalCompilationDatabase &CDB;
  const ThreadsafeFS &TFS;

  // Whether the scanner has scanned the project globally.
  bool GlobalScanned = false;

  clang::tooling::dependencies::DependencyScanningService Service;

  // TODO: Add a scanning cache.

  // Map module name to source file path.
  llvm::StringMap<std::string> ModuleNameToSource;
};

} // namespace clangd
} // namespace clang

#endif
