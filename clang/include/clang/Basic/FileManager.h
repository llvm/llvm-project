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

#ifndef LLVM_CLANG_BASIC_FILEMANAGER_H
#define LLVM_CLANG_BASIC_FILEMANAGER_H

#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/VirtualFileSystem.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include <memory>
#include <map>

namespace llvm {
class MemoryBuffer;
}

namespace clang {
class FileManager;
class FileSystemStatCache;
class PCMCache;

/// \brief Cached information about one directory (either on disk or in
/// the virtual file system).
class DirectoryEntry {
  const char *Name;   // Name of the directory.
  friend class FileManager;
public:
  DirectoryEntry() : Name(nullptr) {}
  const char *getName() const { return Name; }
};

/// \brief Cached information about one file (either on disk
/// or in the virtual file system).
///
/// If the 'File' member is valid, then this FileEntry has an open file
/// descriptor for the file.
class FileEntry {
  const char *Name;           // Name of the file.
  std::string RealPathName;   // Real path to the file; could be empty.
  off_t Size;                 // File size in bytes.
  time_t ModTime;             // Modification time of file.
  const DirectoryEntry *Dir;  // Directory file lives in.
  unsigned UID;               // A unique (small) ID for the file.
  llvm::sys::fs::UniqueID UniqueID;
  bool IsNamedPipe;
  bool InPCH;
  bool IsValid;               // Is this \c FileEntry initialized and valid?

  /// \brief The open file, if it is owned by the \p FileEntry.
  mutable std::unique_ptr<vfs::File> File;
  friend class FileManager;

  void operator=(const FileEntry &) = delete;

public:
  FileEntry()
      : UniqueID(0, 0), IsNamedPipe(false), InPCH(false), IsValid(false)
  {}

  // FIXME: this is here to allow putting FileEntry in std::map.  Once we have
  // emplace, we shouldn't need a copy constructor anymore.
  /// Intentionally does not copy fields that are not set in an uninitialized
  /// \c FileEntry.
  FileEntry(const FileEntry &FE) : UniqueID(FE.UniqueID),
      IsNamedPipe(FE.IsNamedPipe), InPCH(FE.InPCH), IsValid(FE.IsValid) {
    assert(!isValid() && "Cannot copy an initialized FileEntry");
  }

  const char *getName() const { return Name; }
  StringRef tryGetRealPathName() const { return RealPathName; }
  bool isValid() const { return IsValid; }
  off_t getSize() const { return Size; }
  unsigned getUID() const { return UID; }
  const llvm::sys::fs::UniqueID &getUniqueID() const { return UniqueID; }
  bool isInPCH() const { return InPCH; }
  time_t getModificationTime() const { return ModTime; }

  /// \brief Return the directory the file lives in.
  const DirectoryEntry *getDir() const { return Dir; }

  bool operator<(const FileEntry &RHS) const { return UniqueID < RHS.UniqueID; }

  /// \brief Check whether the file is a named pipe (and thus can't be opened by
  /// the native FileManager methods).
  bool isNamedPipe() const { return IsNamedPipe; }

  void closeFile() const {
    File.reset(); // rely on destructor to close File
  }
};

struct FileData;

/// \brief Implements support for file system lookup, file system caching,
/// and directory search management.
///
/// This also handles more advanced properties, such as uniquing files based
/// on "inode", so that a file with two names (e.g. symlinked) will be treated
/// as a single file.
///
class FileManager : public RefCountedBase<FileManager> {
  IntrusiveRefCntPtr<vfs::FileSystem> FS;
  FileSystemOptions FileSystemOpts;

  /// \brief Cache for existing real directories.
  std::map<llvm::sys::fs::UniqueID, DirectoryEntry> UniqueRealDirs;

  /// \brief Cache for existing real files.
  std::map<llvm::sys::fs::UniqueID, FileEntry> UniqueRealFiles;

  /// \brief The virtual directories that we have allocated.
  ///
  /// For each virtual file (e.g. foo/bar/baz.cpp), we add all of its parent
  /// directories (foo/ and foo/bar/) here.
  SmallVector<std::unique_ptr<DirectoryEntry>, 4> VirtualDirectoryEntries;
  /// \brief The virtual files that we have allocated.
  SmallVector<std::unique_ptr<FileEntry>, 4> VirtualFileEntries;

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

  /// \brief The canonical names of directories.
  llvm::DenseMap<const DirectoryEntry *, llvm::StringRef> CanonicalDirNames;

  /// \brief Storage for canonical names that we have computed.
  llvm::BumpPtrAllocator CanonicalNameStorage;

  /// \brief Each FileEntry we create is assigned a unique ID #.
  ///
  unsigned NextFileUID;

  // Statistics.
  unsigned NumDirLookups, NumFileLookups;
  unsigned NumDirCacheMisses, NumFileCacheMisses;

  // Caching.
  std::unique_ptr<FileSystemStatCache> StatCache;

  bool getStatValue(const char *Path, FileData &Data, bool isFile,
                    std::unique_ptr<vfs::File> *F);

  /// Add all ancestors of the given path (pointing to either a file
  /// or a directory) as virtual directories.
  void addAncestorsAsVirtualDirs(StringRef Path);

  /// Manage memory buffers associated with pcm files.
  std::unique_ptr<PCMCache> BufferMgr;

public:
  FileManager(const FileSystemOptions &FileSystemOpts,
              IntrusiveRefCntPtr<vfs::FileSystem> FS = nullptr);
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
  void addStatCache(std::unique_ptr<FileSystemStatCache> statCache,
                    bool AtBeginning = false);

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
  FileSystemOptions &getFileSystemOpts() { return FileSystemOpts; }
  const FileSystemOptions &getFileSystemOpts() const { return FileSystemOpts; }

  IntrusiveRefCntPtr<vfs::FileSystem> getVirtualFileSystem() const {
    return FS;
  }

  /// \brief Retrieve a file entry for a "virtual" file that acts as
  /// if there were a file with the given name on disk.
  ///
  /// The file itself is not accessed.
  const FileEntry *getVirtualFile(StringRef Filename, off_t Size,
                                  time_t ModificationTime);

  /// \brief Open the specified file as a MemoryBuffer, returning a new
  /// MemoryBuffer if successful, otherwise returning null.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  getBufferForFile(const FileEntry *Entry, bool isVolatile = false,
                   bool ShouldCloseOpenFile = true);
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  getBufferForFile(StringRef Filename);

  /// \brief Get the 'stat' information for the given \p Path.
  ///
  /// If the path is relative, it will be resolved against the WorkingDir of the
  /// FileManager's FileSystemOptions.
  ///
  /// \returns false on success, true on error.
  bool getNoncachedStatValue(StringRef Path,
                             vfs::Status &Result);

  /// \brief Remove the real file \p Entry from the cache.
  void invalidateCache(const FileEntry *Entry);

  /// \brief If path is not absolute and FileSystemOptions set the working
  /// directory, the path is modified to be relative to the given
  /// working directory.
  /// \returns true if \c path changed.
  bool FixupRelativePath(SmallVectorImpl<char> &path) const;

  /// Makes \c Path absolute taking into account FileSystemOptions and the
  /// working directory option.
  /// \returns true if \c Path changed to absolute.
  bool makeAbsolutePath(SmallVectorImpl<char> &Path) const;

  /// \brief Produce an array mapping from the unique IDs assigned to each
  /// file to the corresponding FileEntry pointer.
  void GetUniqueIDMapping(
                    SmallVectorImpl<const FileEntry *> &UIDToFiles) const;

  /// \brief Modifies the size and modification time of a previously created
  /// FileEntry. Use with caution.
  static void modifyFileEntry(FileEntry *File, off_t Size,
                              time_t ModificationTime);

  /// \brief Retrieve the canonical name for a given directory.
  ///
  /// This is a very expensive operation, despite its results being cached,
  /// and should only be used when the physical layout of the file system is
  /// required, which is (almost) never.
  StringRef getCanonicalName(const DirectoryEntry *Dir);

  void PrintStats() const;

  /// Return the manager for memory buffers associated with pcm files.
  PCMCache *getPCMCache() {
    return BufferMgr.get();
  }
};

/// This class manages memory buffers associated with pcm files. It also keeps
/// track of the thread context when we start a thread to implicitly build
/// a module.
class PCMCache {
  /// Map a pcm file to a MemoryBuffer and a boolean that tells us whether
  // the pcm file is validated as a system module.
  std::map<std::string, std::pair<std::unique_ptr<llvm::MemoryBuffer>, bool>>
      ConsistentBuffers;

  typedef std::map<std::string, bool> IsSystemMap;
  /// When we start a thread to build a module, we push in a
  /// ModuleCompilationContext; when we end the thread, we pop the
  /// ModuleCompilationContext out.
  struct ModuleCompilationContext {
    /// Keep track of all module files that have been validated in this thread
    /// and whether the module file is validated as a system module.
    IsSystemMap ModulesInParent;
  };
  SmallVector<ModuleCompilationContext, 8> NestedModuleCompilationContexts;

public:
  /// Add a memory buffer for a pcm file to the manager.
  void addConsistentBuffer(std::string FileName,
                           std::unique_ptr<llvm::MemoryBuffer> Buffer);
  /// Remove a pcm file from the map.
  void removeFromConsistentBuffer(std::string FileName);
  /// Return the memory buffer for a given pcm file.
  llvm::MemoryBuffer *lookupConsistentBuffer(std::string Name);
  /// Update the IsSystem boolean flag in ConsistentBuffers. 
  void setIsSystem(std::string FileName, bool IsSystem);

  /// Check if a pcm file was validated by an ancestor, if yes, return
  /// whether it was validated as a system module.
  bool isValidatedByAncestor(std::string FileName, bool &IsSystem);

  void StartCompilation();
  void EndCompilation();

};

}  // end namespace clang

#endif
