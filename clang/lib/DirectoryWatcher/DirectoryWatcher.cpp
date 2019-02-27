//===- DirectoryWatcher.cpp - Listens for directory file changes ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// \brief Utility class for listening for file system changes in a directory.
//===----------------------------------------------------------------------===//

#include "clang/DirectoryWatcher/DirectoryWatcher.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace llvm;

static Optional<sys::fs::file_status> getFileStatus(StringRef path) {
  sys::fs::file_status Status;
  std::error_code EC = status(path, Status);
  if (EC)
    return None;
  return Status;
}

namespace llvm {
// Specialize DenseMapInfo for sys::fs::UniqueID.
template <> struct DenseMapInfo<sys::fs::UniqueID> {
  static sys::fs::UniqueID getEmptyKey() {
    return sys::fs::UniqueID{DenseMapInfo<uint64_t>::getEmptyKey(),
                             DenseMapInfo<uint64_t>::getEmptyKey()};
  }

  static sys::fs::UniqueID getTombstoneKey() {
    return sys::fs::UniqueID{DenseMapInfo<uint64_t>::getTombstoneKey(),
                             DenseMapInfo<uint64_t>::getEmptyKey()};
  }

  static unsigned getHashValue(const sys::fs::UniqueID &val) {
    return DenseMapInfo<std::pair<uint64_t, uint64_t>>::getHashValue(
        std::make_pair(val.getDevice(), val.getFile()));
  }

  static bool isEqual(const sys::fs::UniqueID &LHS,
                      const sys::fs::UniqueID &RHS) {
    return LHS == RHS;
  }
};
} // namespace llvm

namespace {
/// Used for initial directory scan.
///
/// Note that the caller must ensure serial access to it. It is not thread safe
/// to access it without additional protection.
struct DirectoryScan {
  DenseSet<sys::fs::UniqueID> FileIDSet;
  std::vector<std::tuple<std::string, sys::TimePoint<>>> Files;

  void scanDirectory(StringRef Path) {
    using namespace llvm::sys;

    std::error_code EC;
    for (auto It = fs::directory_iterator(Path, EC),
              End = fs::directory_iterator();
         !EC && It != End; It.increment(EC)) {
      auto status = getFileStatus(It->path());
      if (!status.hasValue())
        continue;
      Files.push_back(
          std::make_tuple(It->path(), status->getLastModificationTime()));
      FileIDSet.insert(status->getUniqueID());
    }
  }

  std::vector<DirectoryWatcher::Event> getAsFileEvents() const {
    std::vector<DirectoryWatcher::Event> Events;
    for (const auto &info : Files) {
      DirectoryWatcher::Event Event{DirectoryWatcher::EventKind::Added,
                                    std::get<0>(info), std::get<1>(info)};
      Events.push_back(std::move(Event));
    }
    return Events;
  }
};
} // namespace

// Add platform-specific functionality.

#if !defined(__has_include)
#define __has_include(x) 0
#endif

#if __has_include(<CoreServices/CoreServices.h>)
#include "DirectoryWatcher-mac.inc.h"
#elif __has_include(<sys/inotify.h>)
#include "DirectoryWatcher-linux.inc.h"
#else

struct DirectoryWatcher::Implementation {
  bool initialize(StringRef Path, EventReceiver Receiver, bool waitInitialSync,
                  std::string &Error) {
    Error = "directory listening not supported for this platform";
    return true;
  }
};

#endif

DirectoryWatcher::DirectoryWatcher() : Impl(*new Implementation()) {}

DirectoryWatcher::~DirectoryWatcher() { delete &Impl; }

std::unique_ptr<DirectoryWatcher>
DirectoryWatcher::create(StringRef Path, EventReceiver Receiver,
                         bool waitInitialSync, std::string &Error) {
  using namespace llvm::sys;

  if (!fs::exists(Path)) {
    std::error_code EC = fs::create_directories(Path);
    if (EC) {
      Error = EC.message();
      return nullptr;
    }
  }

  bool IsDir;
  std::error_code EC = fs::is_directory(Path, IsDir);
  if (EC) {
    Error = EC.message();
    return nullptr;
  }
  if (!IsDir) {
    Error = "path is not a directory: ";
    Error += Path;
    return nullptr;
  }

  std::unique_ptr<DirectoryWatcher> DirWatch;
  DirWatch.reset(new DirectoryWatcher());
  auto &Impl = DirWatch->Impl;
  bool hasError =
      Impl.initialize(Path, std::move(Receiver), waitInitialSync, Error);
  if (hasError)
    return nullptr;

  return DirWatch;
}
