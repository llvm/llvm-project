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
#include "Path.h"
#include "clang/Tooling/CompilationDatabase.h"

namespace clang {
namespace clangd {

/// Thread-safe mapping from FileNames to CppFile.
class CppFileCollection {
public:
  std::shared_ptr<CppFile>
  getOrCreateFile(PathRef File, PathRef ResourceDir,
                  GlobalCompilationDatabase &CDB,
                  std::shared_ptr<PCHContainerOperations> PCHs,
                  IntrusiveRefCntPtr<vfs::FileSystem> VFS) {
    std::lock_guard<std::mutex> Lock(Mutex);

    auto It = OpenedFiles.find(File);
    if (It == OpenedFiles.end()) {
      auto Command = getCompileCommand(CDB, File, ResourceDir);

      It = OpenedFiles
               .try_emplace(File, CppFile::Create(File, std::move(Command),
                                                  std::move(PCHs)))
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
  tooling::CompileCommand getCompileCommand(GlobalCompilationDatabase &CDB,
                                            PathRef File, PathRef ResourceDir);

  std::mutex Mutex;
  llvm::StringMap<std::shared_ptr<CppFile>> OpenedFiles;
};
} // namespace clangd
} // namespace clang

#endif
