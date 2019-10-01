//===--- FS.h - File system related utils ------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_FS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_FS_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/VirtualFileSystem.h"

namespace clang {
namespace clangd {

/// Records status information for files open()ed or stat()ed during preamble
/// build (except for the main file), so we can avoid stat()s on the underlying
/// FS when reusing the preamble. For example, code completion can re-stat files
/// when getting FileID for source locations stored in preamble (e.g. checking
/// whether a location is in the main file).
///
/// The cache is keyed by absolute path of file name in cached status, as this
/// is what preamble stores.
///
/// The cache is not thread-safe when updates happen, so the use pattern should
/// be:
///   - One FS writes to the cache from one thread (or several but strictly
///   sequenced), e.g. when building preamble.
///   - Sequence point (no writes after this point, no reads before).
///   - Several FSs can read from the cache, e.g. code completions.
///
/// Note that the cache is only valid when reusing preamble.
class PreambleFileStatusCache {
public:
  /// \p MainFilePath is the absolute path of the main source file this preamble
  /// corresponds to. The stat for the main file will not be cached.
  PreambleFileStatusCache(llvm::StringRef MainFilePath);

  void update(const llvm::vfs::FileSystem &FS, llvm::vfs::Status S);

  /// \p Path is a path stored in preamble.
  llvm::Optional<llvm::vfs::Status> lookup(llvm::StringRef Path) const;

  /// Returns a VFS that collects file status.
  /// Only cache stats for files that exist because
  ///   1) we only care about existing files when reusing preamble, unlike
  ///   building preamble.
  ///   2) we use the file name in the Status as the cache key.
  ///
  /// Note that the returned VFS should not outlive the cache.
  IntrusiveRefCntPtr<llvm::vfs::FileSystem>
  getProducingFS(IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS);

  /// Returns a VFS that uses the cache collected.
  ///
  /// Note that the returned VFS should not outlive the cache.
  IntrusiveRefCntPtr<llvm::vfs::FileSystem>
  getConsumingFS(IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS) const;

private:
  std::string MainFilePath;
  llvm::StringMap<llvm::vfs::Status> StatCache;
};

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_FS_H
