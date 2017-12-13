//===--- ClangdUnitStore.h - A container of CppFiles -------------*-C++-*-===//
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
#include "Logger.h"
#include "Path.h"
#include "clang/Tooling/CompilationDatabase.h"

namespace clang {
namespace clangd {

class Logger;

/// Thread-safe mapping from FileNames to CppFile.
class CppFileCollection {
public:
  std::shared_ptr<CppFile>
  getOrCreateFile(PathRef File, PathRef ResourceDir,
                  GlobalCompilationDatabase &CDB, bool StorePreamblesInMemory,
                  std::shared_ptr<PCHContainerOperations> PCHs) {
    std::lock_guard<std::mutex> Lock(Mutex);

    auto It = OpenedFiles.find(File);
    if (It == OpenedFiles.end()) {
      auto Command = getCompileCommand(CDB, File, ResourceDir);

      It = OpenedFiles
               .try_emplace(File, CppFile::Create(File, std::move(Command),
                                                  StorePreamblesInMemory,
                                                  std::move(PCHs)))
               .first;
    }
    return It->second;
  }

  struct RecreateResult {
    /// A CppFile, stored in this CppFileCollection for the corresponding
    /// filepath after calling recreateFileIfCompileCommandChanged.
    std::shared_ptr<CppFile> FileInCollection;
    /// If a new CppFile had to be created to account for changed
    /// CompileCommand, a previous CppFile instance will be returned in this
    /// field.
    std::shared_ptr<CppFile> RemovedFile;
  };

  /// Similar to getOrCreateFile, but will replace a current CppFile for \p File
  /// with a new one if CompileCommand, provided by \p CDB has changed.
  /// If a currently stored CppFile had to be replaced, the previous instance
  /// will be returned in RecreateResult.RemovedFile.
  RecreateResult recreateFileIfCompileCommandChanged(
      PathRef File, PathRef ResourceDir, GlobalCompilationDatabase &CDB,
      bool StorePreamblesInMemory,
      std::shared_ptr<PCHContainerOperations> PCHs);

  std::shared_ptr<CppFile> getFile(PathRef File) {
    std::lock_guard<std::mutex> Lock(Mutex);

    auto It = OpenedFiles.find(File);
    if (It == OpenedFiles.end())
      return nullptr;
    return It->second;
  }

  /// Removes a CppFile, stored for \p File, if it's inside collection and
  /// returns it.
  std::shared_ptr<CppFile> removeIfPresent(PathRef File);

private:
  tooling::CompileCommand getCompileCommand(GlobalCompilationDatabase &CDB,
                                            PathRef File, PathRef ResourceDir);

  bool compileCommandsAreEqual(tooling::CompileCommand const &LHS,
                               tooling::CompileCommand const &RHS);

  std::mutex Mutex;
  llvm::StringMap<std::shared_ptr<CppFile>> OpenedFiles;
};
} // namespace clangd
} // namespace clang

#endif
