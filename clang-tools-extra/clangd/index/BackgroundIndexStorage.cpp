//== BackgroundIndexStorage.cpp - Provide caching support to BackgroundIndex ==/
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Logger.h"
#include "index/Background.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA1.h"

namespace clang {
namespace clangd {
namespace {

using FileDigest = decltype(llvm::SHA1::hash({}));

static FileDigest digest(StringRef Content) {
  return llvm::SHA1::hash({(const uint8_t *)Content.data(), Content.size()});
}

std::string getShardPathFromFilePath(llvm::StringRef ShardRoot,
                                     llvm::StringRef FilePath) {
  llvm::SmallString<128> ShardRootSS(ShardRoot);
  llvm::sys::path::append(ShardRootSS, llvm::sys::path::filename(FilePath) +
                                           "." + llvm::toHex(digest(FilePath)) +
                                           ".idx");
  return ShardRoot.str();
}

// Uses disk as a storage for index shards. Creates a directory called
// ".clangd-index/" under the path provided during construction.
class DiskBackedIndexStorage : public BackgroundIndexStorage {
  std::string DiskShardRoot;

public:
  // Sets DiskShardRoot to (Directory + ".clangd-index/") which is the base
  // directory for all shard files.
  DiskBackedIndexStorage(llvm::StringRef Directory) {
    llvm::SmallString<128> CDBDirectory(Directory);
    llvm::sys::path::append(CDBDirectory, ".clangd-index/");
    DiskShardRoot = CDBDirectory.str();
    std::error_code OK;
    std::error_code EC = llvm::sys::fs::create_directory(DiskShardRoot);
    if (EC != OK) {
      elog("Failed to create directory {0} for index storage: {1}",
           DiskShardRoot, EC.message());
    }
  }

  std::unique_ptr<IndexFileIn>
  loadShard(llvm::StringRef ShardIdentifier) const override {
    const std::string ShardPath =
        getShardPathFromFilePath(DiskShardRoot, ShardIdentifier);
    auto Buffer = llvm::MemoryBuffer::getFile(ShardPath);
    if (!Buffer)
      return nullptr;
    if (auto I = readIndexFile(Buffer->get()->getBuffer()))
      return llvm::make_unique<IndexFileIn>(std::move(*I));
    else
      elog("Error while reading shard {0}: {1}", ShardIdentifier,
           I.takeError());
    return nullptr;
  }

  llvm::Error storeShard(llvm::StringRef ShardIdentifier,
                         IndexFileOut Shard) const override {
    auto ShardPath = getShardPathFromFilePath(DiskShardRoot, ShardIdentifier);
    std::error_code EC;
    llvm::raw_fd_ostream OS(ShardPath, EC);
    if (EC)
      return llvm::errorCodeToError(EC);
    OS << Shard;
    OS.close();
    return llvm::errorCodeToError(OS.error());
  }
};

// Creates and owns IndexStorages for multiple CDBs.
class DiskBackedIndexStorageManager {
public:
  // Creates or fetches to storage from cache for the specified CDB.
  BackgroundIndexStorage *operator()(llvm::StringRef CDBDirectory) {
    std::lock_guard<std::mutex> Lock(*IndexStorageMapMu);
    auto &IndexStorage = IndexStorageMap[CDBDirectory];
    if (!IndexStorage)
      IndexStorage = llvm::make_unique<DiskBackedIndexStorage>(CDBDirectory);
    return IndexStorage.get();
  }

  // Creates or fetches to storage from cache for the specified CDB.
  BackgroundIndexStorage *createStorage(llvm::StringRef CDBDirectory);

private:
  llvm::StringMap<std::unique_ptr<BackgroundIndexStorage>> IndexStorageMap;
  std::unique_ptr<std::mutex> IndexStorageMapMu;
};

} // namespace

BackgroundIndexStorage::Factory
BackgroundIndexStorage::createDiskBackedStorageFactory() {
  return DiskBackedIndexStorageManager();
}

} // namespace clangd
} // namespace clang
