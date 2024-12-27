//===------------------ ProjectModules.h -------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProjectModules.h"
#include "support/Logger.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningService.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningTool.h"

namespace clang::clangd {
namespace {
/// A scanner to query the dependency information for C++20 Modules.
///
/// The scanner can scan a single file with `scan(PathRef)` member function
/// or scan the whole project with `globalScan(vector<PathRef>)` member
/// function. See the comments of `globalScan` to see the details.
///
/// The ModuleDependencyScanner can get the directly required module names for a
/// specific source file. Also the ModuleDependencyScanner can get the source
/// file declaring the primary module interface for a specific module name.
///
/// IMPORTANT NOTE: we assume that every module unit is only declared once in a
/// source file in the project. But the assumption is not strictly true even
/// besides the invalid projects. The language specification requires that every
/// module unit should be unique in a valid program. But a project can contain
/// multiple programs. Then it is valid that we can have multiple source files
/// declaring the same module in a project as long as these source files don't
/// interfere with each other.
class ModuleDependencyScanner {
public:
  ModuleDependencyScanner(
      std::shared_ptr<const clang::tooling::CompilationDatabase> CDB,
      const ThreadsafeFS &TFS)
      : CDB(CDB), TFS(TFS),
        Service(tooling::dependencies::ScanningMode::CanonicalPreprocessing,
                tooling::dependencies::ScanningOutputFormat::P1689) {}

  /// The scanned modules dependency information for a specific source file.
  struct ModuleDependencyInfo {
    /// The name of the module if the file is a module unit.
    std::optional<std::string> ModuleName;
    /// A list of names for the modules that the file directly depends.
    std::vector<std::string> RequiredModules;
  };

  /// Scanning the single file specified by \param FilePath.
  std::optional<ModuleDependencyInfo> scan(PathRef FilePath);

  /// Scanning every source file in the current project to get the
  /// <module-name> to <module-unit-source> map.
  /// TODO: We should find an efficient method to get the <module-name>
  /// to <module-unit-source> map. We can make it either by providing
  /// a global module dependency scanner to monitor every file. Or we
  /// can simply require the build systems (or even the end users)
  /// to provide the map.
  void globalScan();

  /// Get the source file from the module name. Note that the language
  /// guarantees all the module names are unique in a valid program.
  /// This function should only be called after globalScan.
  ///
  /// TODO: We should handle the case that there are multiple source files
  /// declaring the same module.
  PathRef getSourceForModuleName(llvm::StringRef ModuleName) const;

  /// Return the direct required modules. Indirect required modules are not
  /// included.
  std::vector<std::string> getRequiredModules(PathRef File);

private:
  std::shared_ptr<const clang::tooling::CompilationDatabase> CDB;
  const ThreadsafeFS &TFS;

  // Whether the scanner has scanned the project globally.
  bool GlobalScanned = false;

  clang::tooling::dependencies::DependencyScanningService Service;

  // TODO: Add a scanning cache.

  // Map module name to source file path.
  llvm::StringMap<std::string> ModuleNameToSource;
};

std::optional<ModuleDependencyScanner::ModuleDependencyInfo>
ModuleDependencyScanner::scan(PathRef FilePath) {
  auto Candidates = CDB->getCompileCommands(FilePath);
  if (Candidates.empty())
    return std::nullopt;

  // Choose the first candidates as the compile commands as the file.
  // Following the same logic with
  // DirectoryBasedGlobalCompilationDatabase::getCompileCommand.
  tooling::CompileCommand Cmd = std::move(Candidates.front());

  static int StaticForMainAddr; // Just an address in this process.
  Cmd.CommandLine.push_back("-resource-dir=" +
                            CompilerInvocation::GetResourcesPath(
                                "clangd", (void *)&StaticForMainAddr));

  using namespace clang::tooling::dependencies;

  llvm::SmallString<128> FilePathDir(FilePath);
  llvm::sys::path::remove_filename(FilePathDir);
  DependencyScanningTool ScanningTool(Service, TFS.view(FilePathDir));

  llvm::Expected<P1689Rule> ScanningResult =
      ScanningTool.getP1689ModuleDependencyFile(Cmd, Cmd.Directory);

  if (auto E = ScanningResult.takeError()) {
    elog("Scanning modules dependencies for {0} failed: {1}", FilePath,
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

void ModuleDependencyScanner::globalScan() {
  for (auto &File : CDB->getAllFiles())
    scan(File);

  GlobalScanned = true;
}

PathRef ModuleDependencyScanner::getSourceForModuleName(
    llvm::StringRef ModuleName) const {
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
} // namespace

/// TODO: The existing `ScanningAllProjectModules` is not efficient. See the
/// comments in ModuleDependencyScanner for detail.
///
/// In the future, we wish the build system can provide a well design
/// compilation database for modules then we can query that new compilation
/// database directly. Or we need to have a global long-live scanner to detect
/// the state of each file.
class ScanningAllProjectModules : public ProjectModules {
public:
  ScanningAllProjectModules(
      std::shared_ptr<const clang::tooling::CompilationDatabase> CDB,
      const ThreadsafeFS &TFS)
      : Scanner(CDB, TFS) {}

  ~ScanningAllProjectModules() override = default;

  std::vector<std::string> getRequiredModules(PathRef File) override {
    return Scanner.getRequiredModules(File);
  }

  /// RequiredSourceFile is not used intentionally. See the comments of
  /// ModuleDependencyScanner for detail.
  PathRef
  getSourceForModuleName(llvm::StringRef ModuleName,
                         PathRef RequiredSourceFile = PathRef()) override {
    Scanner.globalScan();
    return Scanner.getSourceForModuleName(ModuleName);
  }

private:
  ModuleDependencyScanner Scanner;
};

std::unique_ptr<ProjectModules> scanningProjectModules(
    std::shared_ptr<const clang::tooling::CompilationDatabase> CDB,
    const ThreadsafeFS &TFS) {
  return std::make_unique<ScanningAllProjectModules>(CDB, TFS);
}

} // namespace clang::clangd
