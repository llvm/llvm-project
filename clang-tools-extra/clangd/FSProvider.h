//===--- FSProvider.h - VFS provider for ClangdServer ------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_FSPROVIDER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_FSPROVIDER_H

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/VirtualFileSystem.h"

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
  virtual llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>
  getFileSystem() const = 0;
};

class RealFileSystemProvider : public FileSystemProvider {
public:
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>
  getFileSystem() const override;
};

} // namespace clangd
} // namespace clang

#endif
