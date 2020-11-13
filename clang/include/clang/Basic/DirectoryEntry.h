//===- clang/Basic/DirectoryEntry.h - Directory references ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines interfaces for clang::DirectoryEntry and clang::DirectoryEntryRef.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_DIRECTORYENTRY_H
#define LLVM_CLANG_BASIC_DIRECTORYENTRY_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorOr.h"

namespace clang {

/// Cached information about one directory (either on disk or in
/// the virtual file system).
class DirectoryEntry {
  friend class FileManager;

  // FIXME: We should not be storing a directory entry name here.
  StringRef Name; // Name of the directory.

public:
  StringRef getName() const { return Name; }
};

/// A reference to a \c DirectoryEntry  that includes the name of the directory
/// as it was accessed by the FileManager's client.
class DirectoryEntryRef {
public:
  const DirectoryEntry &getDirEntry() const { return *Entry->getValue(); }

  StringRef getName() const { return Entry->getKey(); }

private:
  friend class FileManager;

  DirectoryEntryRef(
      llvm::StringMapEntry<llvm::ErrorOr<DirectoryEntry &>> *Entry)
      : Entry(Entry) {}

  const llvm::StringMapEntry<llvm::ErrorOr<DirectoryEntry &>> *Entry;
};

} // end namespace clang

#endif // LLVM_CLANG_BASIC_DIRECTORYENTRY_H
