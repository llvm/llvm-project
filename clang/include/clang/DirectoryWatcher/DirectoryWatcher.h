//===- DirectoryWatcher.h - Listens for directory file changes --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
    /// A file was added to the directory.
    ///
    /// A file gets moved into the directory and replaces an existing file
    /// with the same name will trigger an 'Added' event but no 'Removed' event.
    /// If a file gets replaced multiple times within a short time period, it
    /// may result in only one 'Added' event due to coalescing by the file
    /// system notification mechanism.
    Added,
    /// A file was removed.
    ///
    /// A file that got replaced by another one with the same name will result
    /// in a single 'Added' event, not a 'Removed' one.
    Removed,
    /// A file was modified.
    Modified,
    /// The watched directory got deleted. No more events will follow.
    DirectoryDeleted,
  };

  struct Event {
    EventKind Kind;
    std::string Filename;
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

#endif // LLVM_CLANG_DIRECTORYWATCHER_DIRECTORYWATCHER_H
