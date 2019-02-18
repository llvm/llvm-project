//===--- FS.cpp - File system related utils ----------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "FS.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/None.h"
#include "llvm/Support/Path.h"

using namespace llvm;
namespace clang {
namespace clangd {

PreambleFileStatusCache::PreambleFileStatusCache(StringRef MainFilePath)
    : MainFilePath(MainFilePath) {
  assert(sys::path::is_absolute(MainFilePath));
}

void PreambleFileStatusCache::update(const vfs::FileSystem &FS, vfs::Status S) {
  SmallString<32> PathStore(S.getName());
  if (FS.makeAbsolute(PathStore))
    return;
  // Do not cache status for the main file.
  if (PathStore == MainFilePath)
    return;
  // Stores the latest status in cache as it can change in a preamble build.
  StatCache.insert({PathStore, std::move(S)});
}

Optional<vfs::Status> PreambleFileStatusCache::lookup(StringRef File) const {
  auto I = StatCache.find(File);
  if (I != StatCache.end())
    return I->getValue();
  return None;
}

IntrusiveRefCntPtr<vfs::FileSystem> PreambleFileStatusCache::getProducingFS(
    IntrusiveRefCntPtr<vfs::FileSystem> FS) {
  // This invalidates old status in cache if files are re-`open()`ed or
  // re-`stat()`ed in case file status has changed during preamble build.
  class CollectFS : public vfs::ProxyFileSystem {
  public:
    CollectFS(IntrusiveRefCntPtr<vfs::FileSystem> FS,
              PreambleFileStatusCache &StatCache)
        : ProxyFileSystem(std::move(FS)), StatCache(StatCache) {}

    ErrorOr<std::unique_ptr<vfs::File>>
    openFileForRead(const Twine &Path) override {
      auto File = getUnderlyingFS().openFileForRead(Path);
      if (!File || !*File)
        return File;
      // Eagerly stat opened file, as the followup `status` call on the file
      // doesn't necessarily go through this FS. This puts some extra work on
      // preamble build, but it should be worth it as preamble can be reused
      // many times (e.g. code completion) and the repeated status call is
      // likely to be cached in the underlying file system anyway.
      if (auto S = File->get()->status())
        StatCache.update(getUnderlyingFS(), std::move(*S));
      return File;
    }

    ErrorOr<vfs::Status> status(const Twine &Path) override {
      auto S = getUnderlyingFS().status(Path);
      if (S)
        StatCache.update(getUnderlyingFS(), *S);
      return S;
    }

  private:
    PreambleFileStatusCache &StatCache;
  };
  return IntrusiveRefCntPtr<CollectFS>(new CollectFS(std::move(FS), *this));
}

IntrusiveRefCntPtr<vfs::FileSystem> PreambleFileStatusCache::getConsumingFS(
    IntrusiveRefCntPtr<vfs::FileSystem> FS) const {
  class CacheVFS : public vfs::ProxyFileSystem {
  public:
    CacheVFS(IntrusiveRefCntPtr<vfs::FileSystem> FS,
             const PreambleFileStatusCache &StatCache)
        : ProxyFileSystem(std::move(FS)), StatCache(StatCache) {}

    ErrorOr<vfs::Status> status(const Twine &Path) override {
      if (auto S = StatCache.lookup(Path.str()))
        return *S;
      return getUnderlyingFS().status(Path);
    }

  private:
    const PreambleFileStatusCache &StatCache;
  };
  return IntrusiveRefCntPtr<CacheVFS>(new CacheVFS(std::move(FS), *this));
}

} // namespace clangd
} // namespace clang
