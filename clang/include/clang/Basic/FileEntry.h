//===- clang/Basic/FileEntry.h - File references ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines interfaces for clang::FileEntry and clang::FileEntryRef.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_FILEENTRY_H
#define LLVM_CLANG_BASIC_FILEENTRY_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem/UniqueID.h"

namespace llvm {
namespace vfs {

class File;

} // namespace vfs
} // namespace llvm

namespace clang {

class DirectoryEntry;
class FileEntry;

/// A reference to a \c FileEntry that includes the name of the file as it was
/// accessed by the FileManager's client.
class FileEntryRef {
public:
  const StringRef getName() const { return Entry->first(); }
  const FileEntry &getFileEntry() const {
    return *Entry->second->V.get<FileEntry *>();
  }

  inline bool isValid() const;
  inline off_t getSize() const;
  inline unsigned getUID() const;
  inline const llvm::sys::fs::UniqueID &getUniqueID() const;
  inline time_t getModificationTime() const;

  friend bool operator==(const FileEntryRef &LHS, const FileEntryRef &RHS) {
    return LHS.Entry == RHS.Entry;
  }
  friend bool operator!=(const FileEntryRef &LHS, const FileEntryRef &RHS) {
    return !(LHS == RHS);
  }

  struct MapValue;

  /// Type used in the StringMap.
  using MapEntry = llvm::StringMapEntry<llvm::ErrorOr<MapValue>>;

  /// Type stored in the StringMap.
  struct MapValue {
    /// The pointer at another MapEntry is used when the FileManager should
    /// silently forward from one name to another, which occurs in Redirecting
    /// VFSs that use external names. In that case, the \c FileEntryRef
    /// returned by the \c FileManager will have the external name, and not the
    /// name that was used to lookup the file.
    ///
    /// The second type is really a `const MapEntry *`, but that confuses
    /// gcc5.3.  Once that's no longer supported, change this back.
    llvm::PointerUnion<FileEntry *, const void *> V;

    MapValue() = delete;
    MapValue(FileEntry &FE) : V(&FE) {}
    MapValue(MapEntry &ME) : V(&ME) {}
  };

private:
  friend class FileManager;

  FileEntryRef() = delete;
  explicit FileEntryRef(const MapEntry &Entry)
      : Entry(&Entry) {
    assert(Entry.second && "Expected payload");
    assert(Entry.second->V && "Expected non-null");
    assert(Entry.second->V.is<FileEntry *>() && "Expected FileEntry");
  }

  const MapEntry *Entry;
};

/// Cached information about one file (either on disk
/// or in the virtual file system).
///
/// If the 'File' member is valid, then this FileEntry has an open file
/// descriptor for the file.
class FileEntry {
  friend class FileManager;

  std::string RealPathName;   // Real path to the file; could be empty.
  off_t Size;                 // File size in bytes.
  time_t ModTime;             // Modification time of file.
  const DirectoryEntry *Dir;  // Directory file lives in.
  llvm::sys::fs::UniqueID UniqueID;
  unsigned UID;               // A unique (small) ID for the file.
  bool IsNamedPipe;
  bool IsValid;               // Is this \c FileEntry initialized and valid?

  /// The open file, if it is owned by the \p FileEntry.
  mutable std::unique_ptr<llvm::vfs::File> File;

  // First access name for this FileEntry.
  //
  // This is Optional only to allow delayed construction (FileEntryRef has no
  // default constructor). It should always have a value in practice.
  //
  // TODO: remove this once everyone that needs a name uses FileEntryRef.
  Optional<FileEntryRef> LastRef;

public:
  FileEntry();
  ~FileEntry();

  FileEntry(const FileEntry &) = delete;
  FileEntry &operator=(const FileEntry &) = delete;

  StringRef getName() const { return LastRef->getName(); }
  FileEntryRef getLastRef() const { return *LastRef; }

  StringRef tryGetRealPathName() const { return RealPathName; }
  bool isValid() const { return IsValid; }
  off_t getSize() const { return Size; }
  unsigned getUID() const { return UID; }
  const llvm::sys::fs::UniqueID &getUniqueID() const { return UniqueID; }
  time_t getModificationTime() const { return ModTime; }

  /// Return the directory the file lives in.
  const DirectoryEntry *getDir() const { return Dir; }

  bool operator<(const FileEntry &RHS) const { return UniqueID < RHS.UniqueID; }

  /// Check whether the file is a named pipe (and thus can't be opened by
  /// the native FileManager methods).
  bool isNamedPipe() const { return IsNamedPipe; }

  void closeFile() const;

  // Only for use in tests to see if deferred opens are happening, rather than
  // relying on RealPathName being empty.
  bool isOpenForTests() const { return File != nullptr; }
};

bool FileEntryRef::isValid() const { return getFileEntry().isValid(); }

off_t FileEntryRef::getSize() const { return getFileEntry().getSize(); }

unsigned FileEntryRef::getUID() const { return getFileEntry().getUID(); }

const llvm::sys::fs::UniqueID &FileEntryRef::getUniqueID() const {
  return getFileEntry().getUniqueID();
}

time_t FileEntryRef::getModificationTime() const {
  return getFileEntry().getModificationTime();
}

} // end namespace clang

#endif // LLVM_CLANG_BASIC_FILEENTRY_H
