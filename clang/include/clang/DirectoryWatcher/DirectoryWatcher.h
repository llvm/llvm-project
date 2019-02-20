//===- DirectoryWatcher.h - Listens for directory file changes --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// \brief Utility class for listening for file system changes in a directory.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DIRECTORYWATCHER_DIRECTORYWATCHER_H
#define LLVM_CLANG_DIRECTORYWATCHER_DIRECTORYWATCHER_H

#include "clang/Basic/LLVM.h"
#include "llvm/Support/Chrono.h"
#include <functional>
#include <memory>
#include <string>

namespace clang {

/// Provides notifications for file system changes in a directory.
///
/// Guarantees that the first time the directory is processed, the receiver will
/// be invoked even if the directory is empty.
class DirectoryWatcher {
public:
  enum class EventKind {
    /// A file was added.
    Added,
    /// A file was removed.
    Removed,
    /// A file was modified.
    Modified,
    /// The watched directory got deleted. No more events will follow.
    DirectoryDeleted,
  };

  struct Event {
    EventKind Kind;
    std::string Filename;
    llvm::sys::TimePoint<> ModTime;
  };

  typedef std::function<void(ArrayRef<Event> Events, bool isInitial)>
      EventReceiver;

  ~DirectoryWatcher();

  static std::unique_ptr<DirectoryWatcher> create(StringRef Path,
                                                  EventReceiver Receiver,
                                                  bool waitInitialSync,
                                                  std::string &Error);

private:
  struct Implementation;
  Implementation &Impl;

  DirectoryWatcher();

  DirectoryWatcher(const DirectoryWatcher &) = delete;
  DirectoryWatcher &operator=(const DirectoryWatcher &) = delete;
};

} // namespace clang

#endif
