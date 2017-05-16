//===--- ClangdUnitStore.h - A ClangdUnits container -------------*-C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_CLANGDUNITSTORE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_CLANGDUNITSTORE_H

#include <mutex>

#include "ClangdUnit.h"
#include "GlobalCompilationDatabase.h"
#include "Path.h"
#include "clang/Tooling/CompilationDatabase.h"

namespace clang {
namespace clangd {

/// Thread-safe collection of ASTs built for specific files. Provides
/// synchronized access to ASTs.
class ClangdUnitStore {
public:
  /// Run specified \p Action on the ClangdUnit for \p File.
  /// If the file is not present in ClangdUnitStore, a new ClangdUnit will be
  /// created from the \p FileContents. If the file is already present in the
  /// store, ClangdUnit::reparse will be called with the new contents before
  /// running \p Action.
  template <class Func>
  void runOnUnit(PathRef File, StringRef FileContents,
                 GlobalCompilationDatabase &CDB,
                 std::shared_ptr<PCHContainerOperations> PCHs, Func Action) {
    runOnUnitImpl(File, FileContents, CDB, PCHs, /*ReparseBeforeAction=*/true,
                  std::forward<Func>(Action));
  }

  /// Run specified \p Action on the ClangdUnit for \p File.
  /// If the file is not present in ClangdUnitStore, a new ClangdUnit will be
  /// created from the \p FileContents. If the file is already present in the
  /// store, the \p Action will be run directly on it.
  template <class Func>
  void runOnUnitWithoutReparse(PathRef File, StringRef FileContents,
                               GlobalCompilationDatabase &CDB,
                               std::shared_ptr<PCHContainerOperations> PCHs,
                               Func Action) {
    runOnUnitImpl(File, FileContents, CDB, PCHs, /*ReparseBeforeAction=*/false,
                  std::forward<Func>(Action));
  }

  /// Remove ClangdUnit for \p File, if any
  void removeUnitIfPresent(PathRef File);

private:
  /// Run specified \p Action on the ClangdUnit for \p File.
  template <class Func>
  void runOnUnitImpl(PathRef File, StringRef FileContents,
                     GlobalCompilationDatabase &CDB,
                     std::shared_ptr<PCHContainerOperations> PCHs,
                     bool ReparseBeforeAction, Func Action) {
    std::lock_guard<std::mutex> Lock(Mutex);

    auto Commands = getCompileCommands(CDB, File);
    assert(!Commands.empty() &&
           "getCompileCommands should add default command");
    // chdir. This is thread hostile.
    // FIXME(ibiryukov): get rid of this
    llvm::sys::fs::set_current_path(Commands.front().Directory);

    auto It = OpenedFiles.find(File);
    if (It == OpenedFiles.end()) {
      It = OpenedFiles
               .insert(std::make_pair(
                   File, ClangdUnit(File, FileContents, PCHs, Commands)))
               .first;
    } else if (ReparseBeforeAction) {
      It->second.reparse(FileContents);
    }
    return Action(It->second);
  }

  std::vector<tooling::CompileCommand>
  getCompileCommands(GlobalCompilationDatabase &CDB, PathRef File);

  std::mutex Mutex;
  llvm::StringMap<ClangdUnit> OpenedFiles;
};
} // namespace clangd
} // namespace clang

#endif
