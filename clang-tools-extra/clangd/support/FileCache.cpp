//===--- FileCache.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/FileCache.h"
#include "llvm/ADT/ScopeExit.h"

namespace clang {
namespace clangd {

// Sentinel values for the Size cache key. In both cases, a successful stat of
// the file will never result in the cached value being reused.

// The cached value does not reflect the current content on disk.
static constexpr uint64_t CacheDiskMismatch =
    std::numeric_limits<uint64_t>::max();
// The cached value reflects that the file doesn't exist.
static constexpr uint64_t FileNotFound = CacheDiskMismatch - 1;

FileCache::FileCache(llvm::StringRef Path)
    : Path(Path), ValidTime(std::chrono::steady_clock::time_point::min()),
      ModifiedTime(), Size(CacheDiskMismatch) {
  assert(llvm::sys::path::is_absolute(Path));
}

void FileCache::read(
    const ThreadsafeFS &TFS, std::chrono::steady_clock::time_point FreshTime,
    llvm::function_ref<void(llvm::Optional<llvm::StringRef>)> Parse,
    llvm::function_ref<void()> Read) const {

  std::lock_guard<std::mutex> Lock(Mu);
  // We're going to update the cache and return whatever's in it.
  auto Return = llvm::make_scope_exit(Read);

  // Return any sufficiently recent result without doing any further work.
  if (ValidTime > FreshTime)
    return;

  // Ensure we always bump ValidTime, so that FreshTime imposes a hard limit on
  // how often we do IO.
  auto BumpValidTime = llvm::make_scope_exit(
      [&] { ValidTime = std::chrono::steady_clock::now(); });

  // stat is cheaper than opening the file. It's usually unchanged.
  assert(llvm::sys::path::is_absolute(Path));
  auto FS = TFS.view(/*CWD=*/llvm::None);
  auto Stat = FS->status(Path);
  if (!Stat || !Stat->isRegularFile()) {
    if (Size != FileNotFound) // Allow "not found" value to be cached.
      Parse(llvm::None);
    // Ensure the cache key won't match any future stat().
    Size = FileNotFound;
    return;
  }
  // If the modified-time and size match, assume the content does too.
  if (Size == Stat->getSize() &&
      ModifiedTime == Stat->getLastModificationTime())
    return;

  // OK, the file has actually changed. Update cache key, compute new value.
  Size = Stat->getSize();
  ModifiedTime = Stat->getLastModificationTime();
  // Now read the file from disk.
  if (auto Buf = FS->getBufferForFile(Path)) {
    Parse(Buf->get()->getBuffer());
    // Result is cacheable if the actual read size matches the new cache key.
    // (We can't update the cache key, because we don't know the new mtime).
    if (Buf->get()->getBufferSize() != Size)
      Size = CacheDiskMismatch;
  } else {
    // File was unreadable. Keep the old value and try again next time.
    Size = CacheDiskMismatch;
  }
}

} // namespace clangd
} // namespace clang
