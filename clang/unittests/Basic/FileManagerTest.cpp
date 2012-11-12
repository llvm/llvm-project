//===- unittests/Basic/FileMangerTest.cpp ------------ FileManger tests ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/FileSystemStatCache.h"
#include "clang/Basic/FileManager.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

namespace {

// Used to create a fake file system for running the tests with such
// that the tests are not affected by the structure/contents of the
// file system on the machine running the tests.
class FakeStatCache : public FileSystemStatCache {
private:
  // Maps a file/directory path to its desired stat result.  Anything
  // not in this map is considered to not exist in the file system.
  llvm::StringMap<struct stat, llvm::BumpPtrAllocator> StatCalls;

  void InjectFileOrDirectory(const char *Path, ino_t INode, bool IsFile) {
    struct stat statBuf;
    memset(&statBuf, 0, sizeof(statBuf));
    statBuf.st_dev = 1;
#ifndef _WIN32  // struct stat has no st_ino field on Windows.
    statBuf.st_ino = INode;
#endif
    statBuf.st_mode = IsFile ? (0777 | S_IFREG)  // a regular file
        : (0777 | S_IFDIR);  // a directory
    StatCalls[Path] = statBuf;
  }

public:
  // Inject a file with the given inode value to the fake file system.
  void InjectFile(const char *Path, ino_t INode) {
    InjectFileOrDirectory(Path, INode, /*IsFile=*/true);
  }

  // Inject a directory with the given inode value to the fake file system.
  void InjectDirectory(const char *Path, ino_t INode) {
    InjectFileOrDirectory(Path, INode, /*IsFile=*/false);
  }

  // Implement FileSystemStatCache::getStat().
  virtual LookupResult getStat(const char *Path, struct stat &StatBuf,
                               int *FileDescriptor) {
    if (StatCalls.count(Path) != 0) {
      StatBuf = StatCalls[Path];
      return CacheExists;
    }

    return CacheMissing;  // This means the file/directory doesn't exist.
  }
};

// The test fixture.
class FileManagerTest : public ::testing::Test {
 protected:
  FileManagerTest() : manager(options) {
  }

  FileSystemOptions options;
  FileManager manager;
};

// When a virtual file is added, its getDir() field is set correctly
// (not NULL, correct name).
TEST_F(FileManagerTest, getVirtualFileSetsTheDirFieldCorrectly) {
  const FileEntry *file = manager.getVirtualFile("foo.cpp", 42, 0);
  ASSERT_TRUE(file != NULL);

  const DirectoryEntry *dir = file->getDir();
  ASSERT_TRUE(dir != NULL);
  EXPECT_STREQ(".", dir->getName());

  file = manager.getVirtualFile("x/y/z.cpp", 42, 0);
  ASSERT_TRUE(file != NULL);

  dir = file->getDir();
  ASSERT_TRUE(dir != NULL);
  EXPECT_STREQ("x/y", dir->getName());
}

// Before any virtual file is added, no virtual directory exists.
TEST_F(FileManagerTest, NoVirtualDirectoryExistsBeforeAVirtualFileIsAdded) {
  // An empty FakeStatCache causes all stat calls made by the
  // FileManager to report "file/directory doesn't exist".  This
  // avoids the possibility of the result of this test being affected
  // by what's in the real file system.
  manager.addStatCache(new FakeStatCache);

  EXPECT_EQ(NULL, manager.getDirectory("virtual/dir/foo"));
  EXPECT_EQ(NULL, manager.getDirectory("virtual/dir"));
  EXPECT_EQ(NULL, manager.getDirectory("virtual"));
}

// When a virtual file is added, all of its ancestors should be created.
TEST_F(FileManagerTest, getVirtualFileCreatesDirectoryEntriesForAncestors) {
  // Fake an empty real file system.
  manager.addStatCache(new FakeStatCache);

  manager.getVirtualFile("virtual/dir/bar.h", 100, 0);
  EXPECT_EQ(NULL, manager.getDirectory("virtual/dir/foo"));

  const DirectoryEntry *dir = manager.getDirectory("virtual/dir");
  ASSERT_TRUE(dir != NULL);
  EXPECT_STREQ("virtual/dir", dir->getName());

  dir = manager.getDirectory("virtual");
  ASSERT_TRUE(dir != NULL);
  EXPECT_STREQ("virtual", dir->getName());
}

// getFile() returns non-NULL if a real file exists at the given path.
TEST_F(FileManagerTest, getFileReturnsValidFileEntryForExistingRealFile) {
  // Inject fake files into the file system.
  FakeStatCache *statCache = new FakeStatCache;
  statCache->InjectDirectory("/tmp", 42);
  statCache->InjectFile("/tmp/test", 43);
  manager.addStatCache(statCache);

  const FileEntry *file = manager.getFile("/tmp/test");
  ASSERT_TRUE(file != NULL);
  EXPECT_STREQ("/tmp/test", file->getName());

  const DirectoryEntry *dir = file->getDir();
  ASSERT_TRUE(dir != NULL);
  EXPECT_STREQ("/tmp", dir->getName());
}

// getFile() returns non-NULL if a virtual file exists at the given path.
TEST_F(FileManagerTest, getFileReturnsValidFileEntryForExistingVirtualFile) {
  // Fake an empty real file system.
  manager.addStatCache(new FakeStatCache);

  manager.getVirtualFile("virtual/dir/bar.h", 100, 0);
  const FileEntry *file = manager.getFile("virtual/dir/bar.h");
  ASSERT_TRUE(file != NULL);
  EXPECT_STREQ("virtual/dir/bar.h", file->getName());

  const DirectoryEntry *dir = file->getDir();
  ASSERT_TRUE(dir != NULL);
  EXPECT_STREQ("virtual/dir", dir->getName());
}

// getFile() returns different FileEntries for different paths when
// there's no aliasing.
TEST_F(FileManagerTest, getFileReturnsDifferentFileEntriesForDifferentFiles) {
  // Inject two fake files into the file system.  Different inodes
  // mean the files are not symlinked together.
  FakeStatCache *statCache = new FakeStatCache;
  statCache->InjectDirectory(".", 41);
  statCache->InjectFile("foo.cpp", 42);
  statCache->InjectFile("bar.cpp", 43);
  manager.addStatCache(statCache);

  const FileEntry *fileFoo = manager.getFile("foo.cpp");
  const FileEntry *fileBar = manager.getFile("bar.cpp");
  ASSERT_TRUE(fileFoo != NULL);
  ASSERT_TRUE(fileBar != NULL);
  EXPECT_NE(fileFoo, fileBar);
}

// getFile() returns NULL if neither a real file nor a virtual file
// exists at the given path.
TEST_F(FileManagerTest, getFileReturnsNULLForNonexistentFile) {
  // Inject a fake foo.cpp into the file system.
  FakeStatCache *statCache = new FakeStatCache;
  statCache->InjectDirectory(".", 41);
  statCache->InjectFile("foo.cpp", 42);
  manager.addStatCache(statCache);

  // Create a virtual bar.cpp file.
  manager.getVirtualFile("bar.cpp", 200, 0);

  const FileEntry *file = manager.getFile("xyz.txt");
  EXPECT_EQ(NULL, file);
}

// The following tests apply to Unix-like system only.

#ifndef _WIN32

// getFile() returns the same FileEntry for real files that are aliases.
TEST_F(FileManagerTest, getFileReturnsSameFileEntryForAliasedRealFiles) {
  // Inject two real files with the same inode.
  FakeStatCache *statCache = new FakeStatCache;
  statCache->InjectDirectory("abc", 41);
  statCache->InjectFile("abc/foo.cpp", 42);
  statCache->InjectFile("abc/bar.cpp", 42);
  manager.addStatCache(statCache);

  EXPECT_EQ(manager.getFile("abc/foo.cpp"), manager.getFile("abc/bar.cpp"));
}

// getFile() returns the same FileEntry for virtual files that have
// corresponding real files that are aliases.
TEST_F(FileManagerTest, getFileReturnsSameFileEntryForAliasedVirtualFiles) {
  // Inject two real files with the same inode.
  FakeStatCache *statCache = new FakeStatCache;
  statCache->InjectDirectory("abc", 41);
  statCache->InjectFile("abc/foo.cpp", 42);
  statCache->InjectFile("abc/bar.cpp", 42);
  manager.addStatCache(statCache);

  manager.getVirtualFile("abc/foo.cpp", 100, 0);
  manager.getVirtualFile("abc/bar.cpp", 200, 0);

  EXPECT_EQ(manager.getFile("abc/foo.cpp"), manager.getFile("abc/bar.cpp"));
}

#endif  // !_WIN32

} // anonymous namespace
