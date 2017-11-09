//===- VirtualFileSystem.h - Virtual File System Layer ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// \brief Defines the virtual file system interface vfs::FileSystem.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_VIRTUALFILESYSTEM_H
#define LLVM_CLANG_BASIC_VIRTUALFILESYSTEM_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <ctime>
#include <memory>
#include <stack>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace llvm {

class MemoryBuffer;

} // end namespace llvm

namespace clang {
namespace vfs {

/// \brief The result of a \p status operation.
class Status {
  std::string Name;
  llvm::sys::fs::UniqueID UID;
  llvm::sys::TimePoint<> MTime;
  uint32_t User;
  uint32_t Group;
  uint64_t Size;
  llvm::sys::fs::file_type Type;
  llvm::sys::fs::perms Perms;

public:
  bool IsVFSMapped; // FIXME: remove when files support multiple names

public:
  Status() : Type(llvm::sys::fs::file_type::status_error) {}
  Status(const llvm::sys::fs::file_status &Status);
  Status(StringRef Name, llvm::sys::fs::UniqueID UID,
         llvm::sys::TimePoint<> MTime, uint32_t User, uint32_t Group,
         uint64_t Size, llvm::sys::fs::file_type Type,
         llvm::sys::fs::perms Perms);

  /// Get a copy of a Status with a different name.
  static Status copyWithNewName(const Status &In, StringRef NewName);
  static Status copyWithNewName(const llvm::sys::fs::file_status &In,
                                StringRef NewName);

  /// \brief Returns the name that should be used for this file or directory.
  StringRef getName() const { return Name; }

  /// @name Status interface from llvm::sys::fs
  /// @{
  llvm::sys::fs::file_type getType() const { return Type; }
  llvm::sys::fs::perms getPermissions() const { return Perms; }
  llvm::sys::TimePoint<> getLastModificationTime() const { return MTime; }
  llvm::sys::fs::UniqueID getUniqueID() const { return UID; }
  uint32_t getUser() const { return User; }
  uint32_t getGroup() const { return Group; }
  uint64_t getSize() const { return Size; }
  /// @}
  /// @name Status queries
  /// These are static queries in llvm::sys::fs.
  /// @{
  bool equivalent(const Status &Other) const;
  bool isDirectory() const;
  bool isRegularFile() const;
  bool isOther() const;
  bool isSymlink() const;
  bool isStatusKnown() const;
  bool exists() const;
  /// @}
};

/// \brief Represents an open file.
class File {
public:
  /// \brief Destroy the file after closing it (if open).
  /// Sub-classes should generally call close() inside their destructors.  We
  /// cannot do that from the base class, since close is virtual.
  virtual ~File();

  /// \brief Get the status of the file.
  virtual llvm::ErrorOr<Status> status() = 0;

  /// \brief Get the name of the file
  virtual llvm::ErrorOr<std::string> getName() {
    if (auto Status = status())
      return Status->getName().str();
    else
      return Status.getError();
  }

  /// \brief Get the contents of the file as a \p MemoryBuffer.
  virtual llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  getBuffer(const Twine &Name, int64_t FileSize = -1,
            bool RequiresNullTerminator = true, bool IsVolatile = false) = 0;

  /// \brief Closes the file.
  virtual std::error_code close() = 0;
};

namespace detail {

/// \brief An interface for virtual file systems to provide an iterator over the
/// (non-recursive) contents of a directory.
struct DirIterImpl {
  virtual ~DirIterImpl();

  /// \brief Sets \c CurrentEntry to the next entry in the directory on success,
  /// or returns a system-defined \c error_code.
  virtual std::error_code increment() = 0;

  Status CurrentEntry;
};

} // end namespace detail

/// \brief An input iterator over the entries in a virtual path, similar to
/// llvm::sys::fs::directory_iterator.
class directory_iterator {
  std::shared_ptr<detail::DirIterImpl> Impl; // Input iterator semantics on copy

public:
  directory_iterator(std::shared_ptr<detail::DirIterImpl> I)
      : Impl(std::move(I)) {
    assert(Impl.get() != nullptr && "requires non-null implementation");
    if (!Impl->CurrentEntry.isStatusKnown())
      Impl.reset(); // Normalize the end iterator to Impl == nullptr.
  }

  /// \brief Construct an 'end' iterator.
  directory_iterator() = default;

  /// \brief Equivalent to operator++, with an error code.
  directory_iterator &increment(std::error_code &EC) {
    assert(Impl && "attempting to increment past end");
    EC = Impl->increment();
    if (!Impl->CurrentEntry.isStatusKnown())
      Impl.reset(); // Normalize the end iterator to Impl == nullptr.
    return *this;
  }

  const Status &operator*() const { return Impl->CurrentEntry; }
  const Status *operator->() const { return &Impl->CurrentEntry; }

  bool operator==(const directory_iterator &RHS) const {
    if (Impl && RHS.Impl)
      return Impl->CurrentEntry.equivalent(RHS.Impl->CurrentEntry);
    return !Impl && !RHS.Impl;
  }
  bool operator!=(const directory_iterator &RHS) const {
    return !(*this == RHS);
  }
};

class FileSystem;

/// \brief An input iterator over the recursive contents of a virtual path,
/// similar to llvm::sys::fs::recursive_directory_iterator.
class recursive_directory_iterator {
  typedef std::stack<directory_iterator, std::vector<directory_iterator>>
      IterState;

  FileSystem *FS;
  std::shared_ptr<IterState> State; // Input iterator semantics on copy.

public:
  recursive_directory_iterator(FileSystem &FS, const Twine &Path,
                               std::error_code &EC);
  /// \brief Construct an 'end' iterator.
  recursive_directory_iterator() = default;

  /// \brief Equivalent to operator++, with an error code.
  recursive_directory_iterator &increment(std::error_code &EC);

  const Status &operator*() const { return *State->top(); }
  const Status *operator->() const { return &*State->top(); }

  bool operator==(const recursive_directory_iterator &Other) const {
    return State == Other.State; // identity
  }
  bool operator!=(const recursive_directory_iterator &RHS) const {
    return !(*this == RHS);
  }

  /// \brief Gets the current level. Starting path is at level 0.
  int level() const {
    assert(State->size() && "Cannot get level without any iteration state");
    return State->size()-1;
  }
};

/// \brief The virtual file system interface.
class FileSystem : public llvm::ThreadSafeRefCountedBase<FileSystem> {
public:
  virtual ~FileSystem();

  /// \brief Get the status of the entry at \p Path, if one exists.
  virtual llvm::ErrorOr<Status> status(const Twine &Path) = 0;
  /// \brief Get a \p File object for the file at \p Path, if one exists.
  virtual llvm::ErrorOr<std::unique_ptr<File>>
  openFileForRead(const Twine &Path) = 0;

  /// This is a convenience method that opens a file, gets its content and then
  /// closes the file.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  getBufferForFile(const Twine &Name, int64_t FileSize = -1,
                   bool RequiresNullTerminator = true, bool IsVolatile = false);

  /// \brief Get a directory_iterator for \p Dir.
  /// \note The 'end' iterator is directory_iterator().
  virtual directory_iterator dir_begin(const Twine &Dir,
                                       std::error_code &EC) = 0;

  /// Set the working directory. This will affect all following operations on
  /// this file system and may propagate down for nested file systems.
  virtual std::error_code setCurrentWorkingDirectory(const Twine &Path) = 0;
  /// Get the working directory of this file system.
  virtual llvm::ErrorOr<std::string> getCurrentWorkingDirectory() const = 0;

  /// Check whether a file exists. Provided for convenience.
  bool exists(const Twine &Path);

  /// Make \a Path an absolute path.
  ///
  /// Makes \a Path absolute using the current directory if it is not already.
  /// An empty \a Path will result in the current directory.
  ///
  /// /absolute/path   => /absolute/path
  /// relative/../path => <current-directory>/relative/../path
  ///
  /// \param Path A path that is modified to be an absolute path.
  /// \returns success if \a path has been made absolute, otherwise a
  ///          platform-specific error_code.
  std::error_code makeAbsolute(SmallVectorImpl<char> &Path) const;
};

/// \brief Gets an \p vfs::FileSystem for the 'real' file system, as seen by
/// the operating system.
IntrusiveRefCntPtr<FileSystem> getRealFileSystem();

/// \brief A file system that allows overlaying one \p AbstractFileSystem on top
/// of another.
///
/// Consists of a stack of >=1 \p FileSystem objects, which are treated as being
/// one merged file system. When there is a directory that exists in more than
/// one file system, the \p OverlayFileSystem contains a directory containing
/// the union of their contents.  The attributes (permissions, etc.) of the
/// top-most (most recently added) directory are used.  When there is a file
/// that exists in more than one file system, the file in the top-most file
/// system overrides the other(s).
class OverlayFileSystem : public FileSystem {
  typedef SmallVector<IntrusiveRefCntPtr<FileSystem>, 1> FileSystemList;
  /// \brief The stack of file systems, implemented as a list in order of
  /// their addition.
  FileSystemList FSList;

public:
  OverlayFileSystem(IntrusiveRefCntPtr<FileSystem> Base);
  /// \brief Pushes a file system on top of the stack.
  void pushOverlay(IntrusiveRefCntPtr<FileSystem> FS);

  llvm::ErrorOr<Status> status(const Twine &Path) override;
  llvm::ErrorOr<std::unique_ptr<File>>
  openFileForRead(const Twine &Path) override;
  directory_iterator dir_begin(const Twine &Dir, std::error_code &EC) override;
  llvm::ErrorOr<std::string> getCurrentWorkingDirectory() const override;
  std::error_code setCurrentWorkingDirectory(const Twine &Path) override;

  typedef FileSystemList::reverse_iterator iterator;
  
  /// \brief Get an iterator pointing to the most recently added file system.
  iterator overlays_begin() { return FSList.rbegin(); }

  /// \brief Get an iterator pointing one-past the least recently added file
  /// system.
  iterator overlays_end() { return FSList.rend(); }
};

namespace detail {

class InMemoryDirectory;

} // end namespace detail

/// An in-memory file system.
class InMemoryFileSystem : public FileSystem {
  std::unique_ptr<detail::InMemoryDirectory> Root;
  std::string WorkingDirectory;
  bool UseNormalizedPaths = true;

public:
  explicit InMemoryFileSystem(bool UseNormalizedPaths = true);
  ~InMemoryFileSystem() override;

  /// Add a buffer to the VFS with a path. The VFS owns the buffer.
  /// If present, User, Group, Type and Perms apply to the newly-created file.
  /// \return true if the file was successfully added, false if the file already
  /// exists in the file system with different contents.
  bool addFile(const Twine &Path, time_t ModificationTime,
               std::unique_ptr<llvm::MemoryBuffer> Buffer,
               Optional<uint32_t> User = None, Optional<uint32_t> Group = None,
               Optional<llvm::sys::fs::file_type> Type = None,
               Optional<llvm::sys::fs::perms> Perms = None);
  /// Add a buffer to the VFS with a path. The VFS does not own the buffer.
  /// If present, User, Group, Type and Perms apply to the newly-created file.
  /// \return true if the file was successfully added, false if the file already
  /// exists in the file system with different contents.
  bool addFileNoOwn(const Twine &Path, time_t ModificationTime,
                    llvm::MemoryBuffer *Buffer,
                    Optional<uint32_t> User = None,
                    Optional<uint32_t> Group = None,
                    Optional<llvm::sys::fs::file_type> Type = None,
                    Optional<llvm::sys::fs::perms> Perms = None);

  std::string toString() const;
  /// Return true if this file system normalizes . and .. in paths.
  bool useNormalizedPaths() const { return UseNormalizedPaths; }

  llvm::ErrorOr<Status> status(const Twine &Path) override;
  llvm::ErrorOr<std::unique_ptr<File>>
  openFileForRead(const Twine &Path) override;
  directory_iterator dir_begin(const Twine &Dir, std::error_code &EC) override;
  llvm::ErrorOr<std::string> getCurrentWorkingDirectory() const override {
    return WorkingDirectory;
  }
  std::error_code setCurrentWorkingDirectory(const Twine &Path) override;
};

/// \brief Get a globally unique ID for a virtual file or directory.
llvm::sys::fs::UniqueID getNextVirtualUniqueID();

/// \brief Gets a \p FileSystem for a virtual file system described in YAML
/// format.
IntrusiveRefCntPtr<FileSystem>
getVFSFromYAML(std::unique_ptr<llvm::MemoryBuffer> Buffer,
               llvm::SourceMgr::DiagHandlerTy DiagHandler,
               StringRef YAMLFilePath,
               void *DiagContext = nullptr,
               IntrusiveRefCntPtr<FileSystem> ExternalFS = getRealFileSystem());

struct YAMLVFSEntry {
  template <typename T1, typename T2> YAMLVFSEntry(T1 &&VPath, T2 &&RPath)
      : VPath(std::forward<T1>(VPath)), RPath(std::forward<T2>(RPath)) {}
  std::string VPath;
  std::string RPath;
};

/// \brief Collect all pairs of <virtual path, real path> entries from the
/// \p YAMLFilePath. This is used by the module dependency collector to forward
/// the entries into the reproducer output VFS YAML file.
void collectVFSFromYAML(
    std::unique_ptr<llvm::MemoryBuffer> Buffer,
    llvm::SourceMgr::DiagHandlerTy DiagHandler, StringRef YAMLFilePath,
    SmallVectorImpl<YAMLVFSEntry> &CollectedEntries,
    void *DiagContext = nullptr,
    IntrusiveRefCntPtr<FileSystem> ExternalFS = getRealFileSystem());

class YAMLVFSWriter {
  std::vector<YAMLVFSEntry> Mappings;
  Optional<bool> IsCaseSensitive;
  Optional<bool> IsOverlayRelative;
  Optional<bool> UseExternalNames;
  Optional<bool> IgnoreNonExistentContents;
  std::string OverlayDir;

public:
  YAMLVFSWriter() = default;

  void addFileMapping(StringRef VirtualPath, StringRef RealPath);

  void setCaseSensitivity(bool CaseSensitive) {
    IsCaseSensitive = CaseSensitive;
  }

  void setUseExternalNames(bool UseExtNames) {
    UseExternalNames = UseExtNames;
  }

  void setIgnoreNonExistentContents(bool IgnoreContents) {
    IgnoreNonExistentContents = IgnoreContents;
  }

  void setOverlayDir(StringRef OverlayDirectory) {
    IsOverlayRelative = true;
    OverlayDir.assign(OverlayDirectory.str());
  }

  void write(llvm::raw_ostream &OS);
};

} // end namespace vfs
} // end namespace clang

#endif // LLVM_CLANG_BASIC_VIRTUALFILESYSTEM_H
