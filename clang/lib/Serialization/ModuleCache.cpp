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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LockFileManager.h"
#include "llvm/Support/Path.h"

using namespace clang;

/// Write a new timestamp file with the given path.
static void writeTimestampFile(StringRef TimestampFile) {
  std::error_code EC;
  llvm::raw_fd_ostream Out(TimestampFile.str(), EC, llvm::sys::fs::OF_None);
}

void clang::maybePruneImpl(StringRef Path, time_t PruneInterval,
                           time_t PruneAfter) {
  if (PruneInterval <= 0 || PruneAfter <= 0)
    return;

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
  for (llvm::sys::fs::directory_iterator Dir(Path, EC), DirEnd;
       Dir != DirEnd && !EC; Dir.increment(EC)) {
    // If we don't have a directory, there's nothing to look into.
    if (!llvm::sys::fs::is_directory(Dir->path()))
      continue;

    // Walk all the files within this directory.
    for (llvm::sys::fs::directory_iterator File(Dir->path(), EC), FileEnd;
         File != FileEnd && !EC; File.increment(EC)) {
      // We only care about module and global module index files.
      StringRef Extension = llvm::sys::path::extension(File->path());
      if (Extension != ".pcm" && Extension != ".timestamp" &&
          llvm::sys::path::filename(File->path()) != "modules.idx")
        continue;

      // Look at this file. If we can't stat it, there's nothing interesting
      // there.
      if (llvm::sys::fs::status(File->path(), StatBuf))
        continue;

      // If the file has been used recently enough, leave it there.
      time_t FileAccessTime = llvm::sys::toTimeT(StatBuf.getLastAccessedTime());
      if (CurrentTime - FileAccessTime <= PruneAfter)
        continue;

      // Remove the file.
      llvm::sys::fs::remove(File->path());

      // Remove the timestamp file.
      std::string TimpestampFilename = File->path() + ".timestamp";
      llvm::sys::fs::remove(TimpestampFilename);
    }

    // If we removed all the files in the directory, remove the directory
    // itself.
    if (llvm::sys::fs::directory_iterator(Dir->path(), EC) ==
            llvm::sys::fs::directory_iterator() &&
        !EC)
      llvm::sys::fs::remove(Dir->path());
  }
}

namespace {
class CrossProcessModuleCache : public ModuleCache {
  InMemoryModuleCache InMemory;

public:
  void prepareForGetLock(StringRef ModuleFilename) override {
    // FIXME: Do this in LockFileManager and only if the directory doesn't
    // exist.
    StringRef Dir = llvm::sys::path::parent_path(ModuleFilename);
    llvm::sys::fs::create_directories(Dir);
  }

  std::unique_ptr<llvm::AdvisoryLock>
  getLock(StringRef ModuleFilename) override {
    return std::make_unique<llvm::LockFileManager>(ModuleFilename);
  }

  std::time_t getModuleTimestamp(StringRef ModuleFilename) override {
    std::string TimestampFilename =
        serialization::ModuleFile::getTimestampFilename(ModuleFilename);
    llvm::sys::fs::file_status Status;
    if (llvm::sys::fs::status(TimestampFilename, Status) != std::error_code{})
      return 0;
    return llvm::sys::toTimeT(Status.getLastModificationTime());
  }

  void updateModuleTimestamp(StringRef ModuleFilename) override {
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
    maybePruneImpl(Path, PruneInterval, PruneAfter);
  }

  InMemoryModuleCache &getInMemoryModuleCache() override { return InMemory; }
  const InMemoryModuleCache &getInMemoryModuleCache() const override {
    return InMemory;
  }
};
} // namespace

IntrusiveRefCntPtr<ModuleCache> clang::createCrossProcessModuleCache() {
  return llvm::makeIntrusiveRefCnt<CrossProcessModuleCache>();
}
