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

class FileEntryRef;

} // namespace clang

namespace llvm {
namespace optional_detail {

/// Forward declare a template specialization for OptionalStorage.
template <>
class OptionalStorage<clang::FileEntryRef, /*is_trivially_copyable*/ true>;

} // namespace optional_detail
} // namespace llvm

namespace clang {

class DirectoryEntry;
class FileEntry;

/// A reference to a \c FileEntry that includes the name of the file as it was
/// accessed by the FileManager's client.
class FileEntryRef {
public:
  StringRef getName() const { return ME->first(); }
  const FileEntry &getFileEntry() const {
    return *ME->second->V.get<FileEntry *>();
  }

  inline bool isValid() const;
  inline off_t getSize() const;
  inline unsigned getUID() const;
  inline const llvm::sys::fs::UniqueID &getUniqueID() const;
  inline time_t getModificationTime() const;
  inline void closeFile() const;

  /// Check if the underlying FileEntry is the same, intentially ignoring
  /// whether the file was referenced with the same spelling of the filename.
  friend bool operator==(const FileEntryRef &LHS, const FileEntryRef &RHS) {
    return &LHS.getFileEntry() == &RHS.getFileEntry();
  }
  friend bool operator==(const FileEntry *LHS, const FileEntryRef &RHS) {
    return LHS == &RHS.getFileEntry();
  }
  friend bool operator==(const FileEntryRef &LHS, const FileEntry *RHS) {
    return &LHS.getFileEntry() == RHS;
  }
  friend bool operator!=(const FileEntryRef &LHS, const FileEntryRef &RHS) {
    return !(LHS == RHS);
  }
  friend bool operator!=(const FileEntry *LHS, const FileEntryRef &RHS) {
    return !(LHS == RHS);
  }
  friend bool operator!=(const FileEntryRef &LHS, const FileEntry *RHS) {
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

  /// Check if RHS referenced the file in exactly the same way.
  bool isSameRef(const FileEntryRef &RHS) const { return ME == RHS.ME; }

  /// Allow FileEntryRef to degrade into 'const FileEntry*' to facilitate
  /// incremental adoption.
  ///
  /// The goal is to avoid code churn due to dances like the following:
  /// \code
  /// // Old code.
  /// lvalue = rvalue;
  ///
  /// // Temporary code from an incremental patch.
  /// lvalue = &rvalue.getFileEntry();
  ///
  /// // Final code.
  /// lvalue = rvalue;
  /// \endcode
  ///
  /// FIXME: Once FileEntryRef is "everywhere" and FileEntry::LastRef and
  /// FileEntry::getName have been deleted, delete this implicit conversion.
  operator const FileEntry *() const { return &getFileEntry(); }

  FileEntryRef() = delete;
  explicit FileEntryRef(const MapEntry &ME) : ME(&ME) {
    assert(ME.second && "Expected payload");
    assert(ME.second->V && "Expected non-null");
    assert(ME.second->V.is<FileEntry *>() && "Expected FileEntry");
  }

  /// Expose the underlying MapEntry to simplify packing in a PointerIntPair or
  /// PointerUnion and allow construction in Optional.
  const clang::FileEntryRef::MapEntry &getMapEntry() const { return *ME; }

private:
  friend class llvm::optional_detail::OptionalStorage<
      FileEntryRef, /*is_trivially_copyable*/ true>;
  struct optional_none_tag {};

  // Private constructor for use by OptionalStorage.
  FileEntryRef(optional_none_tag) : ME(nullptr) {}
  bool hasOptionalValue() const { return ME; }

  const MapEntry *ME;
};

static_assert(sizeof(FileEntryRef) == sizeof(const FileEntry *),
              "FileEntryRef must avoid size overhead");

static_assert(std::is_trivially_copyable<FileEntryRef>::value,
              "FileEntryRef must be trivially copyable");

} // end namespace clang

namespace llvm {
namespace optional_detail {

/// Customize OptionalStorage<FileEntryRef> to use FileEntryRef and its
/// optional_none_tag to keep it the size of a single pointer.
template <> class OptionalStorage<clang::FileEntryRef> {
  clang::FileEntryRef MaybeRef;

public:
  ~OptionalStorage() = default;
  OptionalStorage() : MaybeRef(clang::FileEntryRef::optional_none_tag()) {}
  OptionalStorage(OptionalStorage const &Other) = default;
  OptionalStorage(OptionalStorage &&Other) = default;
  OptionalStorage &operator=(OptionalStorage const &Other) = default;
  OptionalStorage &operator=(OptionalStorage &&Other) = default;

  template <class... ArgTypes>
  explicit OptionalStorage(in_place_t, ArgTypes &&...Args)
      : MaybeRef(std::forward<ArgTypes>(Args)...) {}

  void reset() { MaybeRef = clang::FileEntryRef::optional_none_tag(); }

  bool hasValue() const { return MaybeRef.hasOptionalValue(); }

  clang::FileEntryRef &getValue() LLVM_LVALUE_FUNCTION {
    assert(hasValue());
    return MaybeRef;
  }
  clang::FileEntryRef const &getValue() const LLVM_LVALUE_FUNCTION {
    assert(hasValue());
    return MaybeRef;
  }
#if LLVM_HAS_RVALUE_REFERENCE_THIS
  clang::FileEntryRef &&getValue() && {
    assert(hasValue());
    return std::move(MaybeRef);
  }
#endif

  template <class... Args> void emplace(Args &&...args) {
    MaybeRef = clang::FileEntryRef(std::forward<Args>(args)...);
  }

  OptionalStorage &operator=(clang::FileEntryRef Ref) {
    MaybeRef = Ref;
    return *this;
  }
};

static_assert(sizeof(Optional<clang::FileEntryRef>) ==
                  sizeof(clang::FileEntryRef),
              "Optional<FileEntryRef> must avoid size overhead");

static_assert(std::is_trivially_copyable<Optional<clang::FileEntryRef>>::value,
              "Optional<FileEntryRef> should be trivially copyable");

} // end namespace optional_detail
} // end namespace llvm

namespace clang {

/// Wrapper around Optional<FileEntryRef> that degrades to 'const FileEntry*',
/// facilitating incremental patches to propagate FileEntryRef.
///
/// This class can be used as return value or field where it's convenient for
/// an Optional<FileEntryRef> to degrade to a 'const FileEntry*'. The purpose
/// is to avoid code churn due to dances like the following:
/// \code
/// // Old code.
/// lvalue = rvalue;
///
/// // Temporary code from an incremental patch.
/// Optional<FileEntryRef> MaybeF = rvalue;
/// lvalue = MaybeF ? &MaybeF.getFileEntry() : nullptr;
///
/// // Final code.
/// lvalue = rvalue;
/// \endcode
///
/// FIXME: Once FileEntryRef is "everywhere" and FileEntry::LastRef and
/// FileEntry::getName have been deleted, delete this class and replace
/// instances with Optional<FileEntryRef>.
class OptionalFileEntryRefDegradesToFileEntryPtr
    : public Optional<FileEntryRef> {
public:
  OptionalFileEntryRefDegradesToFileEntryPtr() = default;
  OptionalFileEntryRefDegradesToFileEntryPtr(
      OptionalFileEntryRefDegradesToFileEntryPtr &&) = default;
  OptionalFileEntryRefDegradesToFileEntryPtr(
      const OptionalFileEntryRefDegradesToFileEntryPtr &) = default;
  OptionalFileEntryRefDegradesToFileEntryPtr &
  operator=(OptionalFileEntryRefDegradesToFileEntryPtr &&) = default;
  OptionalFileEntryRefDegradesToFileEntryPtr &
  operator=(const OptionalFileEntryRefDegradesToFileEntryPtr &) = default;

  OptionalFileEntryRefDegradesToFileEntryPtr(llvm::NoneType) {}
  OptionalFileEntryRefDegradesToFileEntryPtr(FileEntryRef Ref)
      : Optional<FileEntryRef>(Ref) {}
  OptionalFileEntryRefDegradesToFileEntryPtr(Optional<FileEntryRef> MaybeRef)
      : Optional<FileEntryRef>(MaybeRef) {}

  OptionalFileEntryRefDegradesToFileEntryPtr &operator=(llvm::NoneType) {
    Optional<FileEntryRef>::operator=(None);
    return *this;
  }
  OptionalFileEntryRefDegradesToFileEntryPtr &operator=(FileEntryRef Ref) {
    Optional<FileEntryRef>::operator=(Ref);
    return *this;
  }
  OptionalFileEntryRefDegradesToFileEntryPtr &
  operator=(Optional<FileEntryRef> MaybeRef) {
    Optional<FileEntryRef>::operator=(MaybeRef);
    return *this;
  }

  /// Degrade to 'const FileEntry *' to allow  FileEntry::LastRef and
  /// FileEntry::getName have been deleted, delete this class and replace
  /// instances with Optional<FileEntryRef>
  operator const FileEntry *() const {
    return hasValue() ? &getValue().getFileEntry() : nullptr;
  }
};

static_assert(
    std::is_trivially_copyable<
        OptionalFileEntryRefDegradesToFileEntryPtr>::value,
    "OptionalFileEntryRefDegradesToFileEntryPtr should be trivially copyable");

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

void FileEntryRef::closeFile() const { getFileEntry().closeFile(); }

} // end namespace clang

#endif // LLVM_CLANG_BASIC_FILEENTRY_H
