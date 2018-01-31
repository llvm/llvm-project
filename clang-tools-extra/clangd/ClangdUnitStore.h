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
  explicit CppFileCollection(bool StorePreamblesInMemory,
                             std::shared_ptr<PCHContainerOperations> PCHs,
                             ASTParsedCallback ASTCallback)
      : ASTCallback(std::move(ASTCallback)), PCHs(std::move(PCHs)),
        StorePreamblesInMemory(StorePreamblesInMemory) {}

  std::shared_ptr<CppFile> getOrCreateFile(PathRef File) {
    std::lock_guard<std::mutex> Lock(Mutex);
    auto It = OpenedFiles.find(File);
    if (It == OpenedFiles.end()) {
      It = OpenedFiles
               .try_emplace(File, CppFile::Create(File, StorePreamblesInMemory,
                                                  PCHs, ASTCallback))
               .first;
    }
    return It->second;
  }

  std::shared_ptr<CppFile> getFile(PathRef File) const {
    std::lock_guard<std::mutex> Lock(Mutex);
    auto It = OpenedFiles.find(File);
    if (It == OpenedFiles.end())
      return nullptr;
    return It->second;
  }

  /// Removes a CppFile, stored for \p File, if it's inside collection and
  /// returns it.
  std::shared_ptr<CppFile> removeIfPresent(PathRef File);

  /// Gets used memory for each of the stored files.
  std::vector<std::pair<Path, std::size_t>> getUsedBytesPerFile() const;

private:
  mutable std::mutex Mutex;
  llvm::StringMap<std::shared_ptr<CppFile>> OpenedFiles;
  ASTParsedCallback ASTCallback;
  std::shared_ptr<PCHContainerOperations> PCHs;
  bool StorePreamblesInMemory;
};
} // namespace clangd
} // namespace clang

#endif
