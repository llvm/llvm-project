//===--- ThreadsafeFS.h ------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_SUPPORT_THREADSAFEFS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_SUPPORT_THREADSAFEFS_H

#include "Path.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <memory>

namespace clang {
namespace clangd {

/// Wrapper for vfs::FileSystem for use in multithreaded programs like clangd.
/// As FileSystem is not threadsafe, concurrent threads must each obtain one.
/// Implementations may choose to depend on Context::current() e.g. to implement
/// snapshot semantics. clangd will not create vfs::FileSystems for use in
/// different contexts, so either ThreadsafeFS::view or the returned FS may
/// contain this logic.
class ThreadsafeFS {
public:
  virtual ~ThreadsafeFS() = default;

  /// Obtain a vfs::FileSystem with an arbitrary initial working directory.
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>
  view(llvm::NoneType CWD) const {
    return viewImpl();
  }

  /// Obtain a vfs::FileSystem with a specified working directory.
  /// If the working directory can't be set (e.g. doesn't exist), logs and
  /// returns the FS anyway.
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> view(PathRef CWD) const;

private:
  /// Overridden by implementations to provide a vfs::FileSystem.
  /// This is distinct from view(NoneType) to avoid GCC's -Woverloaded-virtual.
  virtual llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> viewImpl() const = 0;
};

class RealThreadsafeFS : public ThreadsafeFS {
private:
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> viewImpl() const override;
};

} // namespace clangd
} // namespace clang

#endif
