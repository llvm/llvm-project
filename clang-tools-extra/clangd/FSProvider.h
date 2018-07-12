//===--- FSProvider.h - VFS provider for ClangdServer ------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_FSPROVIDER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_FSPROVIDER_H

#include "clang/Basic/VirtualFileSystem.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"

namespace clang {
namespace clangd {

// Wrapper for vfs::FileSystem for use in multithreaded programs like clangd.
// As FileSystem is not threadsafe, concurrent threads must each obtain one.
class FileSystemProvider {
public:
  virtual ~FileSystemProvider() = default;
  /// Called by ClangdServer to obtain a vfs::FileSystem to be used for parsing.
  /// Context::current() will be the context passed to the clang entrypoint,
  /// such as addDocument(), and will also be propagated to result callbacks.
  /// Embedders may use this to isolate filesystem accesses.
  virtual IntrusiveRefCntPtr<vfs::FileSystem> getFileSystem() = 0;
};

class RealFileSystemProvider : public FileSystemProvider {
public:
  // FIXME: returns the single real FS instance, which is not threadsafe.
  IntrusiveRefCntPtr<vfs::FileSystem> getFileSystem() override {
    return vfs::getRealFileSystem();
  }
};

} // namespace clangd
} // namespace clang

#endif
