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
#include "clang/Index/IndexDataStore.h"
#include <functional>
#include <memory>
#include <string>

namespace clang {

/// Provides notifications for file system changes in a directory.
///
/// Guarantees that the first time the directory is processed, the receiver will
/// be invoked even if the directory is empty.
class DirectoryWatcher : public index::AbstractDirectoryWatcher {
  struct Implementation;
  Implementation &Impl;

  DirectoryWatcher();

  DirectoryWatcher(const DirectoryWatcher&) = delete;
  DirectoryWatcher &operator =(const DirectoryWatcher&) = delete;

public:
  ~DirectoryWatcher();

  static std::unique_ptr<DirectoryWatcher>
    create(StringRef Path, EventReceiver Receiver, bool waitInitialSync,
           std::string &Error);
};

} // namespace clang

#endif
