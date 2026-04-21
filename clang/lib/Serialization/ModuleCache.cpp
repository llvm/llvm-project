//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Serialization/ModuleCache.h"

#include "clang/Serialization/InMemoryModuleCache.h"
#include "clang/Serialization/ModuleFile.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/IOSandbox.h"
#include "llvm/Support/LockFileManager.h"
#include "llvm/Support/Path.h"

using namespace clang;

const ModuleCacheDirectory *ModuleCache::getDirectoryPtr(StringRef Path) {
  auto [ByNameIt, ByNameInserted] = ByPath.insert({Path, nullptr});
  if (!ByNameIt->second) {
    // This is a compiler-internal input/output, let's bypass the sandbox.
    auto BypassSandbox = llvm::sys::sandbox::scopedDisable();
    llvm::sys::fs::file_status Status;
    if (llvm::sys::fs::status(Path, Status))
      return nullptr;
    llvm::sys::fs::UniqueID UID = Status.getUniqueID();
    auto [ByUIDIt, ByUIDInserted] = ByUID.insert({UID, nullptr});
    if (!ByUIDIt->second)
      ByUIDIt->second = std::make_unique<ModuleCacheDirectory>();
    ByNameIt->second = ByUIDIt->second.get();
  }
  return ByNameIt->second;
}

/// Write a new timestamp file with the given path.
static void writeTimestampFile(StringRef TimestampFile) {
  std::error_code EC;
  llvm::raw_fd_ostream Out(TimestampFile.str(), EC, llvm::sys::fs::OF_None);
}

void clang::maybePruneImpl(StringRef Path, time_t PruneInterval,
                           time_t PruneAfter, bool PruneTopLevel) {
  if (PruneInterval <= 0 || PruneAfter <= 0)
    return;

  // This is a compiler-internal input/output, let's bypass the sandbox.
  auto BypassSandbox = llvm::sys::sandbox::scopedDisable();

  llvm::SmallString<128> TimestampFile(Path);
  llvm::sys::path::append(TimestampFile, "modules.timestamp");

  // Try to stat() the timestamp file.
  llvm::sys::fs::file_status StatBuf;
  if (std::error_code EC = llvm::sys::fs::status(TimestampFile, StatBuf)) {
    // If the timestamp file wasn't there, create one now.
    if (EC == std::errc::no_such_file_or_directory)
      writeTimestampFile(TimestampFile);
    return;
  }

  // Check whether the time stamp is older than our pruning interval.
  // If not, do nothing.
  time_t TimestampModTime =
      llvm::sys::toTimeT(StatBuf.getLastModificationTime());
  time_t CurrentTime = time(nullptr);
  if (CurrentTime - TimestampModTime <= PruneInterval)
    return;

  // Write a new timestamp file so that nobody else attempts to prune.
  // There is a benign race condition here, if two Clang instances happen to
  // notice at the same time that the timestamp is out-of-date.
  writeTimestampFile(TimestampFile);

  // Walk the entire module cache, looking for unused module files and module
  // indices.
  std::error_code EC;
  auto TryPruneFile = [&](StringRef FilePath) {
    // We only care about module and global module index files.
    StringRef Filename = llvm::sys::path::filename(FilePath);
    StringRef Extension = llvm::sys::path::extension(FilePath);
    if (Extension != ".pcm" && Extension != ".timestamp" &&
        Filename != "modules.idx")
      return;

    // Don't prune the pruning timestamp file.
    if (Filename == "modules.timestamp")
      return;

    // Look at this file. If we can't stat it, there's nothing interesting
    // there.
    if (llvm::sys::fs::status(FilePath, StatBuf))
      return;

    // If the file has been used recently enough, leave it there.
    time_t FileAccessTime = llvm::sys::toTimeT(StatBuf.getLastAccessedTime());
    if (CurrentTime - FileAccessTime <= PruneAfter)
      return;

    // Remove the file.
    llvm::sys::fs::remove(FilePath);

    // Remove the timestamp file created by implicit module builds.
    std::string TimestampFilename = FilePath.str() + ".timestamp";
    llvm::sys::fs::remove(TimestampFilename);
  };

  for (llvm::sys::fs::directory_iterator Dir(Path, EC), DirEnd;
       Dir != DirEnd && !EC; Dir.increment(EC)) {
    // If we don't have a directory, try to prune it as a file in the root.
    if (!llvm::sys::fs::is_directory(Dir->path())) {
      if (PruneTopLevel)
        TryPruneFile(Dir->path());
      continue;
    }

    // Walk all the files within this directory.
    for (llvm::sys::fs::directory_iterator File(Dir->path(), EC), FileEnd;
         File != FileEnd && !EC; File.increment(EC))
      TryPruneFile(File->path());

    // If we removed all the files in the directory, remove the directory
    // itself.
    if (llvm::sys::fs::directory_iterator(Dir->path(), EC) ==
            llvm::sys::fs::directory_iterator() &&
        !EC)
      llvm::sys::fs::remove(Dir->path());
  }
}

std::error_code clang::writeImpl(StringRef Path, llvm::MemoryBufferRef Buffer,
                                 off_t &Size, time_t &ModTime) {
  StringRef Extension = llvm::sys::path::extension(Path);
  SmallString<128> ModelPath = StringRef(Path).drop_back(Extension.size());
  ModelPath += "-%%%%%%%%";
  ModelPath += Extension;
  ModelPath += ".tmp";

  std::error_code EC;
  int FD;
  SmallString<128> TmpPath;
  if ((EC = llvm::sys::fs::createUniqueFile(ModelPath, FD, TmpPath))) {
    if (EC != std::errc::no_such_file_or_directory)
      return EC;

    StringRef Dir = llvm::sys::path::parent_path(Path);
    if (std::error_code InnerEC = llvm::sys::fs::create_directories(Dir))
      return InnerEC;

    if ((EC = llvm::sys::fs::createUniqueFile(ModelPath, FD, TmpPath)))
      return EC;
  }

  llvm::sys::fs::file_status Status;
  {
    llvm::raw_fd_ostream OS(FD, /*shouldClose=*/true);
    OS << Buffer.getBuffer();
    // Using the status from an open file descriptor ensures this is not racy.
    if ((EC = llvm::sys::fs::status(FD, Status)))
      return EC;
  }

  Size = Status.getSize();
  ModTime = llvm::sys::toTimeT(Status.getLastModificationTime());

  // This preserves both size and modification time.
  if ((EC = llvm::sys::fs::rename(TmpPath, Path)))
    return EC;

  return {};
}

Expected<std::unique_ptr<llvm::MemoryBuffer>>
clang::readImpl(StringRef FileName, off_t &Size, time_t &ModTime) {
  Expected<llvm::sys::fs::file_t> FD =
      llvm::sys::fs::openNativeFileForRead(FileName);
  if (!FD)
    return FD.takeError();
  llvm::scope_exit CloseFD([&FD]() { llvm::sys::fs::closeFile(*FD); });
  llvm::sys::fs::file_status Status;
  if (std::error_code EC = llvm::sys::fs::status(*FD, Status))
    return llvm::errorCodeToError(EC);
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Buf =
      llvm::MemoryBuffer::getOpenFile(*FD, FileName, Status.getSize(),
                                      /*RequiresNullTerminator=*/false);
  if (!Buf)
    return llvm::errorCodeToError(Buf.getError());
  Size = Status.getSize();
  ModTime = llvm::sys::toTimeT(Status.getLastModificationTime());
  return std::move(*Buf);
}

namespace {
class CrossProcessModuleCache : public ModuleCache {
  InMemoryModuleCache InMemory;

public:
  std::unique_ptr<llvm::AdvisoryLock>
  getLock(StringRef ModuleFilename) override {
    return std::make_unique<llvm::LockFileManager>(ModuleFilename);
  }

  std::time_t getModuleTimestamp(StringRef ModuleFilename) override {
    // This is a compiler-internal input/output, let's bypass the sandbox.
    auto BypassSandbox = llvm::sys::sandbox::scopedDisable();

    std::string TimestampFilename =
        serialization::ModuleFile::getTimestampFilename(ModuleFilename);
    llvm::sys::fs::file_status Status;
    if (llvm::sys::fs::status(TimestampFilename, Status) != std::error_code{})
      return 0;
    return llvm::sys::toTimeT(Status.getLastModificationTime());
  }

  void updateModuleTimestamp(StringRef ModuleFilename) override {
    // This is a compiler-internal input/output, let's bypass the sandbox.
    auto BypassSandbox = llvm::sys::sandbox::scopedDisable();

    // Overwrite the timestamp file contents so that file's mtime changes.
    std::error_code EC;
    llvm::raw_fd_ostream OS(
        serialization::ModuleFile::getTimestampFilename(ModuleFilename), EC,
        llvm::sys::fs::OF_TextWithCRLF);
    if (EC)
      return;
    OS << "Timestamp file\n";
    OS.close();
    OS.clear_error(); // Avoid triggering a fatal error.
  }

  void maybePrune(StringRef Path, time_t PruneInterval,
                  time_t PruneAfter) override {
    // This is a compiler-internal input/output, let's bypass the sandbox.
    auto BypassSandbox = llvm::sys::sandbox::scopedDisable();

    maybePruneImpl(Path, PruneInterval, PruneAfter);
  }

  InMemoryModuleCache &getInMemoryModuleCache() override { return InMemory; }
  const InMemoryModuleCache &getInMemoryModuleCache() const override {
    return InMemory;
  }

  std::error_code write(StringRef Path, llvm::MemoryBufferRef Buffer,
                        off_t &Size, time_t &ModTime) override {
    // This is a compiler-internal input/output, let's bypass the sandbox.
    auto BypassSandbox = llvm::sys::sandbox::scopedDisable();

    return writeImpl(Path, Buffer, Size, ModTime);
  }

  Expected<std::unique_ptr<llvm::MemoryBuffer>>
  read(StringRef FileName, off_t &Size, time_t &ModTime) override {
    // This is a compiler-internal input/output, let's bypass the sandbox.
    auto BypassSandbox = llvm::sys::sandbox::scopedDisable();

    return readImpl(FileName, Size, ModTime);
  }
};
} // namespace

std::shared_ptr<ModuleCache> clang::createCrossProcessModuleCache() {
  return std::make_shared<CrossProcessModuleCache>();
}
