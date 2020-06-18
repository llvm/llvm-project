//===--- FileManager.h - File System Probing and Caching --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines the clang::FileManager interface and associated types.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_FILEMANAGER_H
#define LLVM_CLANG_BASIC_FILEMANAGER_H

#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <ctime>
#include <map>
#include <memory>
#include <string>

namespace llvm {

class MemoryBuffer;

} // end namespace llvm

namespace clang {

class FileSystemStatCache;

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

/// Cached information about one file (either on disk
/// or in the virtual file system).
///
/// If the 'File' member is valid, then this FileEntry has an open file
/// descriptor for the file.
class FileEntry {
  friend class FileManager;

  StringRef Name;             // Name of the file.
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

public:
  FileEntry()
      : UniqueID(0, 0), IsNamedPipe(false), IsValid(false)
  {}

  FileEntry(const FileEntry &) = delete;
  FileEntry &operator=(const FileEntry &) = delete;

  StringRef getName() const { return Name; }
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

  void closeFile() const {
    File.reset(); // rely on destructor to close File
  }

  // Only for use in tests to see if deferred opens are happening, rather than
  // relying on RealPathName being empty.
  bool isOpenForTests() const { return File != nullptr; }
};

/// A reference to a \c FileEntry that includes the name of the file as it was
/// accessed by the FileManager's client.
class FileEntryRef {
public:
  FileEntryRef() = delete;
  FileEntryRef(StringRef Name, const FileEntry &Entry)
      : Name(Name), Entry(&Entry) {}

  const StringRef getName() const { return Name; }

  bool isValid() const { return Entry->isValid(); }

  const FileEntry &getFileEntry() const { return *Entry; }

  off_t getSize() const { return Entry->getSize(); }

  unsigned getUID() const { return Entry->getUID(); }

  const llvm::sys::fs::UniqueID &getUniqueID() const {
    return Entry->getUniqueID();
  }

  time_t getModificationTime() const { return Entry->getModificationTime(); }

  friend bool operator==(const FileEntryRef &LHS, const FileEntryRef &RHS) {
    return LHS.Entry == RHS.Entry && LHS.Name == RHS.Name;
  }
  friend bool operator!=(const FileEntryRef &LHS, const FileEntryRef &RHS) {
    return !(LHS == RHS);
  }

private:
  StringRef Name;
  const FileEntry *Entry;
};

/// Implements support for file system lookup, file system caching,
/// and directory search management.
///
/// This also handles more advanced properties, such as uniquing files based
/// on "inode", so that a file with two names (e.g. symlinked) will be treated
/// as a single file.
///
class FileManager : public RefCountedBase<FileManager> {
  IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS;
  FileSystemOptions FileSystemOpts;

  /// Cache for existing real directories.
  std::map<llvm::sys::fs::UniqueID, DirectoryEntry> UniqueRealDirs;

  /// Cache for existing real files.
  std::map<llvm::sys::fs::UniqueID, FileEntry> UniqueRealFiles;

  /// The virtual directories that we have allocated.
  ///
  /// For each virtual file (e.g. foo/bar/baz.cpp), we add all of its parent
  /// directories (foo/ and foo/bar/) here.
  SmallVector<std::unique_ptr<DirectoryEntry>, 4> VirtualDirectoryEntries;
  /// The virtual files that we have allocated.
  SmallVector<std::unique_ptr<FileEntry>, 4> VirtualFileEntries;

  /// A set of files that bypass the maps and uniquing.  They can have
  /// conflicting filenames.
  SmallVector<std::unique_ptr<FileEntry>, 0> BypassFileEntries;

  /// A cache that maps paths to directory entries (either real or
  /// virtual) we have looked up, or an error that occurred when we looked up
  /// the directory.
  ///
  /// The actual Entries for real directories/files are
  /// owned by UniqueRealDirs/UniqueRealFiles above, while the Entries
  /// for virtual directories/files are owned by
  /// VirtualDirectoryEntries/VirtualFileEntries above.
  ///
  llvm::StringMap<llvm::ErrorOr<DirectoryEntry &>, llvm::BumpPtrAllocator>
  SeenDirEntries;

  /// A reference to the file entry that is associated with a particular
  /// filename, or a reference to another filename that should be looked up
  /// instead of the accessed filename.
  ///
  /// The reference to another filename is specifically useful for Redirecting
  /// VFSs that use external names. In that case, the \c FileEntryRef returned
  /// by the \c FileManager will have the external name, and not the name that
  /// was used to lookup the file.
  using SeenFileEntryOrRedirect =
      llvm::PointerUnion<FileEntry *, const StringRef *>;

  /// A cache that maps paths to file entries (either real or
  /// virtual) we have looked up, or an error that occurred when we looked up
  /// the file.
  ///
  /// \see SeenDirEntries
  llvm::StringMap<llvm::ErrorOr<SeenFileEntryOrRedirect>,
                  llvm::BumpPtrAllocator>
      SeenFileEntries;

  /// The canonical names of files and directories .
  llvm::DenseMap<const void *, llvm::StringRef> CanonicalNames;

  /// Storage for canonical names that we have computed.
  llvm::BumpPtrAllocator CanonicalNameStorage;

  /// Each FileEntry we create is assigned a unique ID #.
  ///
  unsigned NextFileUID;

  // Caching.
  std::unique_ptr<FileSystemStatCache> StatCache;

  std::error_code getStatValue(StringRef Path, llvm::vfs::Status &Status,
                               bool isFile,
                               std::unique_ptr<llvm::vfs::File> *F);

  /// Add all ancestors of the given path (pointing to either a file
  /// or a directory) as virtual directories.
  void addAncestorsAsVirtualDirs(StringRef Path);

  /// Fills the RealPathName in file entry.
  void fillRealPathName(FileEntry *UFE, llvm::StringRef FileName);

public:
  /// Construct a file manager, optionally with a custom VFS.
  ///
  /// \param FS if non-null, the VFS to use.  Otherwise uses
  /// llvm::vfs::getRealFileSystem().
  FileManager(const FileSystemOptions &FileSystemOpts,
              IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS = nullptr);
  ~FileManager();

  /// Installs the provided FileSystemStatCache object within
  /// the FileManager.
  ///
  /// Ownership of this object is transferred to the FileManager.
  ///
  /// \param statCache the new stat cache to install. Ownership of this
  /// object is transferred to the FileManager.
  void setStatCache(std::unique_ptr<FileSystemStatCache> statCache);

  /// Removes the FileSystemStatCache object from the manager.
  void clearStatCache();

  /// Returns the number of unique real file entries cached by the file manager.
  size_t getNumUniqueRealFiles() const { return UniqueRealFiles.size(); }

  /// Lookup, cache, and verify the specified directory (real or
  /// virtual).
  ///
  /// This returns a \c std::error_code if there was an error reading the
  /// directory. On success, returns the reference to the directory entry
  /// together with the exact path that was used to access a file by a
  /// particular call to getDirectoryRef.
  ///
  /// \param CacheFailure If true and the file does not exist, we'll cache
  /// the failure to find this file.
  llvm::Expected<DirectoryEntryRef> getDirectoryRef(StringRef DirName,
                                                    bool CacheFailure = true);

  /// Get a \c DirectoryEntryRef if it exists, without doing anything on error.
  llvm::Optional<DirectoryEntryRef>
  getOptionalDirectoryRef(StringRef DirName, bool CacheFailure = true) {
    return llvm::expectedToOptional(getDirectoryRef(DirName, CacheFailure));
  }

  /// Lookup, cache, and verify the specified directory (real or
  /// virtual).
  ///
  /// This function is deprecated and will be removed at some point in the
  /// future, new clients should use
  ///  \c getDirectoryRef.
  ///
  /// This returns a \c std::error_code if there was an error reading the
  /// directory. If there is no error, the DirectoryEntry is guaranteed to be
  /// non-NULL.
  ///
  /// \param CacheFailure If true and the file does not exist, we'll cache
  /// the failure to find this file.
  llvm::ErrorOr<const DirectoryEntry *>
  getDirectory(StringRef DirName, bool CacheFailure = true);

  /// Lookup, cache, and verify the specified file (real or
  /// virtual).
  ///
  /// This function is deprecated and will be removed at some point in the
  /// future, new clients should use
  ///  \c getFileRef.
  ///
  /// This returns a \c std::error_code if there was an error loading the file.
  /// If there is no error, the FileEntry is guaranteed to be non-NULL.
  ///
  /// \param OpenFile if true and the file exists, it will be opened.
  ///
  /// \param CacheFailure If true and the file does not exist, we'll cache
  /// the failure to find this file.
  llvm::ErrorOr<const FileEntry *>
  getFile(StringRef Filename, bool OpenFile = false, bool CacheFailure = true);

  /// Lookup, cache, and verify the specified file (real or virtual). Return the
  /// reference to the file entry together with the exact path that was used to
  /// access a file by a particular call to getFileRef. If the underlying VFS is
  /// a redirecting VFS that uses external file names, the returned FileEntryRef
  /// will use the external name instead of the filename that was passed to this
  /// method.
  ///
  /// This returns a \c std::error_code if there was an error loading the file,
  /// or a \c FileEntryRef otherwise.
  ///
  /// \param OpenFile if true and the file exists, it will be opened.
  ///
  /// \param CacheFailure If true and the file does not exist, we'll cache
  /// the failure to find this file.
  llvm::Expected<FileEntryRef> getFileRef(StringRef Filename,
                                          bool OpenFile = false,
                                          bool CacheFailure = true);

  /// Get a FileEntryRef if it exists, without doing anything on error.
  llvm::Optional<FileEntryRef> getOptionalFileRef(StringRef Filename,
                                                  bool OpenFile = false,
                                                  bool CacheFailure = true) {
    return llvm::expectedToOptional(
        getFileRef(Filename, OpenFile, CacheFailure));
  }

  /// Returns the current file system options
  FileSystemOptions &getFileSystemOpts() { return FileSystemOpts; }
  const FileSystemOptions &getFileSystemOpts() const { return FileSystemOpts; }

  llvm::vfs::FileSystem &getVirtualFileSystem() const { return *FS; }

  void setVirtualFileSystem(IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS) {
    this->FS = std::move(FS);
  }

  /// Retrieve a file entry for a "virtual" file that acts as
  /// if there were a file with the given name on disk.
  ///
  /// The file itself is not accessed.
  const FileEntry *getVirtualFile(StringRef Filename, off_t Size,
                                  time_t ModificationTime);

  /// Retrieve a FileEntry that bypasses VFE, which is expected to be a virtual
  /// file entry, to access the real file.  The returned FileEntry will have
  /// the same filename as FE but a different identity and its own stat.
  ///
  /// This should be used only for rare error recovery paths because it
  /// bypasses all mapping and uniquing, blindly creating a new FileEntry.
  /// There is no attempt to deduplicate these; if you bypass the same file
  /// twice, you get two new file entries.
  llvm::Optional<FileEntryRef> getBypassFile(FileEntryRef VFE);

  /// Open the specified file as a MemoryBuffer, returning a new
  /// MemoryBuffer if successful, otherwise returning null.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  getBufferForFile(const FileEntry *Entry, bool isVolatile = false,
                   bool RequiresNullTerminator = true);
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  getBufferForFile(StringRef Filename, bool isVolatile = false,
                   bool RequiresNullTerminator = true) {
    return getBufferForFileImpl(Filename, /*FileSize=*/-1, isVolatile,
                                RequiresNullTerminator);
  }

private:
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  getBufferForFileImpl(StringRef Filename, int64_t FileSize, bool isVolatile,
                       bool RequiresNullTerminator);

public:
  /// Get the 'stat' information for the given \p Path.
  ///
  /// If the path is relative, it will be resolved against the WorkingDir of the
  /// FileManager's FileSystemOptions.
  ///
  /// \returns a \c std::error_code describing an error, if there was one
  std::error_code getNoncachedStatValue(StringRef Path,
                                        llvm::vfs::Status &Result);

  /// If path is not absolute and FileSystemOptions set the working
  /// directory, the path is modified to be relative to the given
  /// working directory.
  /// \returns true if \c path changed.
  bool FixupRelativePath(SmallVectorImpl<char> &path) const;

  /// Makes \c Path absolute taking into account FileSystemOptions and the
  /// working directory option.
  /// \returns true if \c Path changed to absolute.
  bool makeAbsolutePath(SmallVectorImpl<char> &Path) const;

  /// Produce an array mapping from the unique IDs assigned to each
  /// file to the corresponding FileEntry pointer.
  void GetUniqueIDMapping(
                    SmallVectorImpl<const FileEntry *> &UIDToFiles) const;

  /// Retrieve the canonical name for a given directory.
  ///
  /// This is a very expensive operation, despite its results being cached,
  /// and should only be used when the physical layout of the file system is
  /// required, which is (almost) never.
  StringRef getCanonicalName(const DirectoryEntry *Dir);

  /// Retrieve the canonical name for a given file.
  ///
  /// This is a very expensive operation, despite its results being cached,
  /// and should only be used when the physical layout of the file system is
  /// required, which is (almost) never.
  StringRef getCanonicalName(const FileEntry *File);

  void PrintStats() const;
};

} // end namespace clang

#endif // LLVM_CLANG_BASIC_FILEMANAGER_H
