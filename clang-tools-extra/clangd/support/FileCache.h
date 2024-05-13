//===--- FileCache.h - Revalidating cache of data from disk ------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_SUPPORT_FILECACHE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_SUPPORT_FILECACHE_H

#include "Path.h"
#include "ThreadsafeFS.h"
#include "llvm/Support/Chrono.h"
#include <mutex>
#include <optional>

namespace clang {
namespace clangd {

/// Base class for threadsafe cache of data read from a file on disk.
///
/// We want configuration files to be "live" as much as possible.
/// Reading them every time is simplest, but caching solves a few problems:
///  - reading and parsing is cheap but not free (and happens on hot paths)
///  - we can ignore invalid data and use the old value (we may see truncated
///    compile_commands.json from non-atomic writers)
///  - we avoid reporting the same errors repeatedly
///
/// We still read and parse the data synchronously on demand, but skip as much
/// work as possible:
///  - if not enough wall-time has elapsed, assume the data is still up-to-date
///  - if we stat the file and it has the same mtime + size, don't read it
///  - obviously we only have to parse when we re-read the file
/// (Tracking OS change events is an alternative, but difficult to do portably.)
///
/// Caches for particular data (e.g. compilation databases) should inherit and:
///  - add mutable storage for the cached parsed data
///  - add a public interface implemented on top of read()
class FileCache {
protected:
  // Path must be absolute.
  FileCache(PathRef Path);

  // Updates the cached value if needed, then provides threadsafe access to it.
  //
  // Specifically:
  // - Parse() may be called (if the cache was not up-to-date)
  //   The lock is held, so cache storage may be safely written.
  //   Parse(None) means the file doesn't exist.
  // - Read() will always be called, to provide access to the value.
  //   The lock is again held, so the value can be copied or used.
  //
  // If the last Parse is newer than FreshTime, we don't check metadata.
  //   - time_point::min() means we only do IO if we never read the file before
  //   - time_point::max() means we always at least stat the file
  //   - steady_clock::now() + seconds(1) means we accept 1 second of staleness
  void read(const ThreadsafeFS &TFS,
            std::chrono::steady_clock::time_point FreshTime,
            llvm::function_ref<void(std::optional<llvm::StringRef>)> Parse,
            llvm::function_ref<void()> Read) const;

  PathRef path() const { return Path; }

private:
  std::string Path;
  // Members are mutable so read() can present a const interface.
  // (It is threadsafe and approximates read-through to TFS).
  mutable std::mutex Mu;
  // Time when the cache was known valid (reflected disk state).
  mutable std::chrono::steady_clock::time_point ValidTime;
  // Filesystem metadata corresponding to the currently cached data.
  mutable llvm::sys::TimePoint<> ModifiedTime;
  mutable uint64_t Size;
};

} // namespace clangd
} // namespace clang

#endif
