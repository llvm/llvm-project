//===--- FSProvider.h - VFS provider for ClangdServer ------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_SUPPORT_FSPROVIDER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_SUPPORT_FSPROVIDER_H

#include "Path.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <memory>

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
  /// Initial working directory is arbitrary.
  virtual llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>
  getFileSystem(llvm::NoneType CWD) const = 0;

  /// As above, except it will try to set current working directory to \p CWD.
  /// This is an overload instead of an optional to make implicit string ->
  /// StringRef conversion possible.
  virtual llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>
  getFileSystem(PathRef CWD) const;
};

class RealFileSystemProvider : public FileSystemProvider {
public:
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>
      getFileSystem(llvm::NoneType) const override;
};

} // namespace clangd
} // namespace clang

#endif
