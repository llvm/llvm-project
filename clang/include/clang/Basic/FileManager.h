//===--- FileManager.h - File System Probing and Caching --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines the clang::FileManager interface and associated types.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FILEMANAGER_H
#define LLVM_CLANG_FILEMANAGER_H

#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/Allocator.h"
// FIXME: Enhance libsystem to support inode and other fields in stat.
#include <sys/types.h>

#ifdef _MSC_VER
typedef unsigned short mode_t;
#endif

struct stat;

namespace llvm {
class MemoryBuffer;
namespace sys { class Path; }
}

namespace clang {
class FileManager;
class FileSystemStatCache;

/// \brief Cached information about one directory (either on disk or in
/// the virtual file system).
class DirectoryEntry {
  const char *Name;   // Name of the directory.
  friend class FileManager;
public:
  DirectoryEntry() : Name(0) {}
  const char *getName() const { return Name; }
};

/// \brief Cached information about one file (either on disk
/// or in the virtual file system).
///
/// If the 'FD' member is valid, then this FileEntry has an open file
/// descriptor for the file.
class FileEntry {
  const char *Name;           // Name of the file.
  off_t Size;                 // File size in bytes.
  time_t ModTime;             // Modification time of file.
  const DirectoryEntry *Dir;  // Directory file lives in.
  unsigned UID;               // A unique (small) ID for the file.
  dev_t Device;               // ID for the device containing the file.
  ino_t Inode;                // Inode number for the file.
  mode_t FileMode;            // The file mode as returned by 'stat'.

  /// FD - The file descriptor for the file entry if it is opened and owned
  /// by the FileEntry.  If not, this is set to -1.
  mutable int FD;
  friend class FileManager;

public:
  FileEntry(dev_t device, ino_t inode, mode_t m)
    : Name(0), Device(device), Inode(inode), FileMode(m), FD(-1) {}
  // Add a default constructor for use with llvm::StringMap
  FileEntry() : Name(0), Device(0), Inode(0), FileMode(0), FD(-1) {}

  FileEntry(const FileEntry &FE) {
    memcpy(this, &FE, sizeof(FE));
    assert(FD == -1 && "Cannot copy a file-owning FileEntry");
  }

  void operator=(const FileEntry &FE) {
    memcpy(this, &FE, sizeof(FE));
    assert(FD == -1 && "Cannot assign a file-owning FileEntry");
  }

  ~FileEntry();

  const char *getName() const { return Name; }
  off_t getSize() const { return Size; }
  unsigned getUID() const { return UID; }
  ino_t getInode() const { return Inode; }
  dev_t getDevice() const { return Device; }
  time_t getModificationTime() const { return ModTime; }
  mode_t getFileMode() const { return FileMode; }

  /// \brief Return the directory the file lives in.
  const DirectoryEntry *getDir() const { return Dir; }

  bool operator<(const FileEntry &RHS) const {
    return Device < RHS.Device || (Device == RHS.Device && Inode < RHS.Inode);
  }

  /// \brief Check whether the file is a named pipe (and thus can't be opened by
  /// the native FileManager methods).
  bool isNamedPipe() const;
};

/// \brief Implements support for file system lookup, file system caching,
/// and directory search management.
///
/// This also handles more advanced properties, such as uniquing files based
/// on "inode", so that a file with two names (e.g. symlinked) will be treated
/// as a single file.
///
class FileManager : public RefCountedBase<FileManager> {
  FileSystemOptions FileSystemOpts;

  class UniqueDirContainer;
  class UniqueFileContainer;

  /// \brief Cache for existing real directories.
  UniqueDirContainer &UniqueRealDirs;

  /// \brief Cache for existing real files.
  UniqueFileContainer &UniqueRealFiles;

  /// \brief The virtual directories that we have allocated.
  ///
  /// For each virtual file (e.g. foo/bar/baz.cpp), we add all of its parent
  /// directories (foo/ and foo/bar/) here.
  SmallVector<DirectoryEntry*, 4> VirtualDirectoryEntries;
  /// \brief The virtual files that we have allocated.
  SmallVector<FileEntry*, 4> VirtualFileEntries;

  /// \brief A cache that maps paths to directory entries (either real or
  /// virtual) we have looked up
  ///
  /// The actual Entries for real directories/files are
  /// owned by UniqueRealDirs/UniqueRealFiles above, while the Entries
  /// for virtual directories/files are owned by
  /// VirtualDirectoryEntries/VirtualFileEntries above.
  ///
  llvm::StringMap<DirectoryEntry*, llvm::BumpPtrAllocator> SeenDirEntries;

  /// \brief A cache that maps paths to file entries (either real or
  /// virtual) we have looked up.
  ///
  /// \see SeenDirEntries
  llvm::StringMap<FileEntry*, llvm::BumpPtrAllocator> SeenFileEntries;

  /// \brief Each FileEntry we create is assigned a unique ID #.
  ///
  unsigned NextFileUID;

  // Statistics.
  unsigned NumDirLookups, NumFileLookups;
  unsigned NumDirCacheMisses, NumFileCacheMisses;

  // Caching.
  OwningPtr<FileSystemStatCache> StatCache;

  bool getStatValue(const char *Path, struct stat &StatBuf,
                    int *FileDescriptor);

  /// Add all ancestors of the given path (pointing to either a file
  /// or a directory) as virtual directories.
  void addAncestorsAsVirtualDirs(StringRef Path);

public:
  FileManager(const FileSystemOptions &FileSystemOpts);
  ~FileManager();

  /// \brief Installs the provided FileSystemStatCache object within
  /// the FileManager.
  ///
  /// Ownership of this object is transferred to the FileManager.
  ///
  /// \param statCache the new stat cache to install. Ownership of this
  /// object is transferred to the FileManager.
  ///
  /// \param AtBeginning whether this new stat cache must be installed at the
  /// beginning of the chain of stat caches. Otherwise, it will be added to
  /// the end of the chain.
  void addStatCache(FileSystemStatCache *statCache, bool AtBeginning = false);

  /// \brief Removes the specified FileSystemStatCache object from the manager.
  void removeStatCache(FileSystemStatCache *statCache);

  /// \brief Removes all FileSystemStatCache objects from the manager.
  void clearStatCaches();

  /// \brief Lookup, cache, and verify the specified directory (real or
  /// virtual).
  ///
  /// This returns NULL if the directory doesn't exist.
  ///
  /// \param CacheFailure If true and the file does not exist, we'll cache
  /// the failure to find this file.
  const DirectoryEntry *getDirectory(StringRef DirName,
                                     bool CacheFailure = true);

  /// \brief Lookup, cache, and verify the specified file (real or
  /// virtual).
  ///
  /// This returns NULL if the file doesn't exist.
  ///
  /// \param OpenFile if true and the file exists, it will be opened.
  ///
  /// \param CacheFailure If true and the file does not exist, we'll cache
  /// the failure to find this file.
  const FileEntry *getFile(StringRef Filename, bool OpenFile = false,
                           bool CacheFailure = true);

  /// \brief Returns the current file system options
  const FileSystemOptions &getFileSystemOptions() { return FileSystemOpts; }

  /// \brief Retrieve a file entry for a "virtual" file that acts as
  /// if there were a file with the given name on disk.
  ///
  /// The file itself is not accessed.
  const FileEntry *getVirtualFile(StringRef Filename, off_t Size,
                                  time_t ModificationTime);

  /// \brief Open the specified file as a MemoryBuffer, returning a new
  /// MemoryBuffer if successful, otherwise returning null.
  llvm::MemoryBuffer *getBufferForFile(const FileEntry *Entry,
                                       std::string *ErrorStr = 0,
                                       bool isVolatile = false);
  llvm::MemoryBuffer *getBufferForFile(StringRef Filename,
                                       std::string *ErrorStr = 0);

  /// \brief Get the 'stat' information for the given \p Path.
  ///
  /// If the path is relative, it will be resolved against the WorkingDir of the
  /// FileManager's FileSystemOptions.
  bool getNoncachedStatValue(StringRef Path, struct stat &StatBuf);

  /// \brief Remove the real file \p Entry from the cache.
  void invalidateCache(const FileEntry *Entry);

  /// \brief If path is not absolute and FileSystemOptions set the working
  /// directory, the path is modified to be relative to the given
  /// working directory.
  void FixupRelativePath(SmallVectorImpl<char> &path) const;

  /// \brief Produce an array mapping from the unique IDs assigned to each
  /// file to the corresponding FileEntry pointer.
  void GetUniqueIDMapping(
                    SmallVectorImpl<const FileEntry *> &UIDToFiles) const;

  /// \brief Modifies the size and modification time of a previously created
  /// FileEntry. Use with caution.
  static void modifyFileEntry(FileEntry *File, off_t Size,
                              time_t ModificationTime);

  void PrintStats() const;
};

}  // end namespace clang

#endif
