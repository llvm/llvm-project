//===------------------ ProjectModules.h -------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProjectModules.h"

namespace clang {
namespace clangd {

/// TODO: The existing `ScanningAllProjectModules` is not efficient. See the
/// comments in ModuleDependencyScanner for detail.
///
/// In the future, we wish the build system can provide a well design
/// compilation database for modules then we can query that new compilation
/// database directly. Or we need to have a global long-live scanner to detect
/// the state of each file.
class ScanningAllProjectModules : public ProjectModules {
public:
  ScanningAllProjectModules(std::vector<std::string> &&AllFiles,
                            const GlobalCompilationDatabase &CDB,
                            const ThreadsafeFS &TFS)
      : AllFiles(std::move(AllFiles)), Scanner(CDB, TFS) {}

  ~ScanningAllProjectModules() override = default;

  std::vector<std::string> getRequiredModules(PathRef File) override {
    return Scanner.getRequiredModules(File);
  }

  /// RequiredSourceFile is not used intentionally. See the comments of
  /// ModuleDependencyScanner for detail.
  PathRef
  getSourceForModuleName(StringRef ModuleName,
                         PathRef RequiredSourceFile = PathRef()) override {
    if (!Scanner.isGlobalScanned())
      Scanner.globalScan(AllFiles);

    return Scanner.getSourceForModuleName(ModuleName);
  }

private:
  std::vector<std::string> AllFiles;

  ModuleDependencyScanner Scanner;
};

std::shared_ptr<ProjectModules> ProjectModules::create(
    ProjectModulesKind Kind, std::vector<std::string> &&AllFiles,
    const GlobalCompilationDatabase &CDB, const ThreadsafeFS &TFS) {
  if (Kind == ProjectModulesKind::ScanningAllFiles)
    return std::make_shared<ScanningAllProjectModules>(std::move(AllFiles), CDB,
                                                       TFS);

  llvm_unreachable("Unknown ProjectModulesKind.");
}

} // namespace clangd
} // namespace clang