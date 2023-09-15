//===---------------- ModuleDependencyScanner.cpp ----------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ModuleDependencyScanner.h"
#include "support/Logger.h"

namespace clang {
namespace clangd {

std::optional<ModuleDependencyScanner::ModuleDependencyInfo>
ModuleDependencyScanner::scan(PathRef FilePath) {
  std::optional<tooling::CompileCommand> Cmd = CDB.getCompileCommand(FilePath);

  if (!Cmd)
    return std::nullopt;

  using namespace clang::tooling::dependencies;

  llvm::SmallString<128> FilePathDir(FilePath);
  llvm::sys::path::remove_filename(FilePathDir);
  DependencyScanningTool ScanningTool(Service, TFS.view(FilePathDir));

  llvm::Expected<P1689Rule> ScanningResult =
      ScanningTool.getP1689ModuleDependencyFile(*Cmd, Cmd->Directory);

  if (auto E = ScanningResult.takeError()) {
    log("Scanning modules dependencies for {0} failed: {1}", FilePath,
        llvm::toString(std::move(E)));
    return std::nullopt;
  }

  ModuleDependencyInfo Result;

  if (ScanningResult->Provides) {
    ModuleNameToSource[ScanningResult->Provides->ModuleName] = FilePath;
    Result.ModuleName = ScanningResult->Provides->ModuleName;
  }

  for (auto &Required : ScanningResult->Requires)
    Result.RequiredModules.push_back(Required.ModuleName);

  return Result;
}

void ModuleDependencyScanner::globalScan(
    const std::vector<std::string> &AllFiles) {
  for (auto &File : AllFiles)
    scan(File);

  GlobalScanned = true;
}

PathRef
ModuleDependencyScanner::getSourceForModuleName(StringRef ModuleName) const {
  assert(
      GlobalScanned &&
      "We should only call getSourceForModuleName after calling globalScan()");

  if (auto It = ModuleNameToSource.find(ModuleName);
      It != ModuleNameToSource.end())
    return It->second;

  return {};
}

std::vector<std::string>
ModuleDependencyScanner::getRequiredModules(PathRef File) {
  auto ScanningResult = scan(File);
  if (!ScanningResult)
    return {};

  return ScanningResult->RequiredModules;
}

} // namespace clangd
} // namespace clang
