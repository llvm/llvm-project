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

namespace clang {
namespace clangd {

PreambleFileStatusCache::PreambleFileStatusCache(llvm::StringRef MainFilePath)
    : MainFilePath(MainFilePath) {
  assert(llvm::sys::path::is_absolute(MainFilePath));
}

void PreambleFileStatusCache::update(const llvm::vfs::FileSystem &FS,
                                     llvm::vfs::Status S) {
  llvm::SmallString<32> PathStore(S.getName());
  if (FS.makeAbsolute(PathStore))
    return;
  // Do not cache status for the main file.
  if (PathStore == MainFilePath)
    return;
  // Stores the latest status in cache as it can change in a preamble build.
  StatCache.insert({PathStore, std::move(S)});
}

llvm::Optional<llvm::vfs::Status>
PreambleFileStatusCache::lookup(llvm::StringRef File) const {
  auto I = StatCache.find(File);
  if (I != StatCache.end())
    return I->getValue();
  return None;
}

llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>
PreambleFileStatusCache::getProducingFS(
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS) {
  // This invalidates old status in cache if files are re-`open()`ed or
  // re-`stat()`ed in case file status has changed during preamble build.
  class CollectFS : public llvm::vfs::ProxyFileSystem {
  public:
    CollectFS(llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS,
              PreambleFileStatusCache &StatCache)
        : ProxyFileSystem(std::move(FS)), StatCache(StatCache) {}

    llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>>
    openFileForRead(const llvm::Twine &Path) override {
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

    llvm::ErrorOr<llvm::vfs::Status> status(const llvm::Twine &Path) override {
      auto S = getUnderlyingFS().status(Path);
      if (S)
        StatCache.update(getUnderlyingFS(), *S);
      return S;
    }

  private:
    PreambleFileStatusCache &StatCache;
  };
  return llvm::IntrusiveRefCntPtr<CollectFS>(
      new CollectFS(std::move(FS), *this));
}

llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>
PreambleFileStatusCache::getConsumingFS(
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS) const {
  class CacheVFS : public llvm::vfs::ProxyFileSystem {
  public:
    CacheVFS(llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS,
             const PreambleFileStatusCache &StatCache)
        : ProxyFileSystem(std::move(FS)), StatCache(StatCache) {}

    llvm::ErrorOr<llvm::vfs::Status> status(const llvm::Twine &Path) override {
      if (auto S = StatCache.lookup(Path.str()))
        return *S;
      return getUnderlyingFS().status(Path);
    }

  private:
    const PreambleFileStatusCache &StatCache;
  };
  return llvm::IntrusiveRefCntPtr<CacheVFS>(new CacheVFS(std::move(FS), *this));
}

} // namespace clangd
} // namespace clang
