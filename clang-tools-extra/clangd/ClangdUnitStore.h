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

#include "ClangdUnit.h"
#include "GlobalCompilationDatabase.h"
#include "Logger.h"
#include "Path.h"
#include "clang/Tooling/CompilationDatabase.h"
#include <mutex>

namespace clang {
namespace clangd {

class Logger;

/// Thread-safe mapping from FileNames to CppFile.
class CppFileCollection {
public:
  /// \p ASTCallback is called when a file is parsed synchronously. This should
  /// not be expensive since it blocks diagnostics.
  explicit CppFileCollection(ASTParsedCallback ASTCallback)
      : ASTCallback(std::move(ASTCallback)) {}

  std::shared_ptr<CppFile>
  getOrCreateFile(PathRef File, PathRef ResourceDir,
                  bool StorePreamblesInMemory,
                  std::shared_ptr<PCHContainerOperations> PCHs) {
    std::lock_guard<std::mutex> Lock(Mutex);
    auto It = OpenedFiles.find(File);
    if (It == OpenedFiles.end()) {
      It = OpenedFiles
               .try_emplace(File, CppFile::Create(File, StorePreamblesInMemory,
                                                  std::move(PCHs), ASTCallback))
               .first;
    }
    return It->second;
  }

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
  std::mutex Mutex;
  llvm::StringMap<std::shared_ptr<CppFile>> OpenedFiles;
  ASTParsedCallback ASTCallback;
};
} // namespace clangd
} // namespace clang

#endif
