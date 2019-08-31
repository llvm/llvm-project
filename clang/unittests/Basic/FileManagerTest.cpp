//===- unittests/Basic/FileMangerTest.cpp ------------ FileManger tests ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/FileSystemStatCache.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"
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
  llvm::StringMap<llvm::vfs::Status, llvm::BumpPtrAllocator> StatCalls;

  void InjectFileOrDirectory(const char *Path, ino_t INode, bool IsFile) {
#ifndef _WIN32
    SmallString<128> NormalizedPath(Path);
    llvm::sys::path::native(NormalizedPath);
    Path = NormalizedPath.c_str();
#endif

    auto fileType = IsFile ?
      llvm::sys::fs::file_type::regular_file :
      llvm::sys::fs::file_type::directory_file;
    llvm::vfs::Status Status(Path, llvm::sys::fs::UniqueID(1, INode),
                             /*MTime*/{}, /*User*/0, /*Group*/0,
                             /*Size*/0, fileType,
                             llvm::sys::fs::perms::all_all);
    StatCalls[Path] = Status;
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
  std::error_code getStat(StringRef Path, llvm::vfs::Status &Status,
                          bool isFile,
                          std::unique_ptr<llvm::vfs::File> *F,
                          llvm::vfs::FileSystem &FS) override {
#ifndef _WIN32
    SmallString<128> NormalizedPath(Path);
    llvm::sys::path::native(NormalizedPath);
    Path = NormalizedPath.c_str();
#endif

    if (StatCalls.count(Path) != 0) {
      Status = StatCalls[Path];
      return std::error_code();
    }

    return std::make_error_code(std::errc::no_such_file_or_directory);
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
  ASSERT_TRUE(file != nullptr);

  const DirectoryEntry *dir = file->getDir();
  ASSERT_TRUE(dir != nullptr);
  EXPECT_EQ(".", dir->getName());

  file = manager.getVirtualFile("x/y/z.cpp", 42, 0);
  ASSERT_TRUE(file != nullptr);

  dir = file->getDir();
  ASSERT_TRUE(dir != nullptr);
  EXPECT_EQ("x/y", dir->getName());
}

// Before any virtual file is added, no virtual directory exists.
TEST_F(FileManagerTest, NoVirtualDirectoryExistsBeforeAVirtualFileIsAdded) {
  // An empty FakeStatCache causes all stat calls made by the
  // FileManager to report "file/directory doesn't exist".  This
  // avoids the possibility of the result of this test being affected
  // by what's in the real file system.
  manager.setStatCache(std::make_unique<FakeStatCache>());

  ASSERT_FALSE(manager.getDirectory("virtual/dir/foo"));
  ASSERT_FALSE(manager.getDirectory("virtual/dir"));
  ASSERT_FALSE(manager.getDirectory("virtual"));
}

// When a virtual file is added, all of its ancestors should be created.
TEST_F(FileManagerTest, getVirtualFileCreatesDirectoryEntriesForAncestors) {
  // Fake an empty real file system.
  manager.setStatCache(std::make_unique<FakeStatCache>());

  manager.getVirtualFile("virtual/dir/bar.h", 100, 0);
  ASSERT_FALSE(manager.getDirectory("virtual/dir/foo"));

  auto dir = manager.getDirectory("virtual/dir");
  ASSERT_TRUE(dir);
  EXPECT_EQ("virtual/dir", (*dir)->getName());

  dir = manager.getDirectory("virtual");
  ASSERT_TRUE(dir);
  EXPECT_EQ("virtual", (*dir)->getName());
}

// getFile() returns non-NULL if a real file exists at the given path.
TEST_F(FileManagerTest, getFileReturnsValidFileEntryForExistingRealFile) {
  // Inject fake files into the file system.
  auto statCache = std::make_unique<FakeStatCache>();
  statCache->InjectDirectory("/tmp", 42);
  statCache->InjectFile("/tmp/test", 43);

#ifdef _WIN32
  const char *DirName = "C:.";
  const char *FileName = "C:test";
  statCache->InjectDirectory(DirName, 44);
  statCache->InjectFile(FileName, 45);
#endif

  manager.setStatCache(std::move(statCache));

  auto file = manager.getFile("/tmp/test");
  ASSERT_TRUE(file);
  ASSERT_TRUE((*file)->isValid());
  EXPECT_EQ("/tmp/test", (*file)->getName());

  const DirectoryEntry *dir = (*file)->getDir();
  ASSERT_TRUE(dir != nullptr);
  EXPECT_EQ("/tmp", dir->getName());

#ifdef _WIN32
  file = manager.getFile(FileName);
  ASSERT_TRUE(file);

  dir = (*file)->getDir();
  ASSERT_TRUE(dir != NULL);
  EXPECT_EQ(DirName, dir->getName());
#endif
}

// getFile() returns non-NULL if a virtual file exists at the given path.
TEST_F(FileManagerTest, getFileReturnsValidFileEntryForExistingVirtualFile) {
  // Fake an empty real file system.
  manager.setStatCache(std::make_unique<FakeStatCache>());

  manager.getVirtualFile("virtual/dir/bar.h", 100, 0);
  auto file = manager.getFile("virtual/dir/bar.h");
  ASSERT_TRUE(file);
  ASSERT_TRUE((*file)->isValid());
  EXPECT_EQ("virtual/dir/bar.h", (*file)->getName());

  const DirectoryEntry *dir = (*file)->getDir();
  ASSERT_TRUE(dir != nullptr);
  EXPECT_EQ("virtual/dir", dir->getName());
}

// getFile() returns different FileEntries for different paths when
// there's no aliasing.
TEST_F(FileManagerTest, getFileReturnsDifferentFileEntriesForDifferentFiles) {
  // Inject two fake files into the file system.  Different inodes
  // mean the files are not symlinked together.
  auto statCache = std::make_unique<FakeStatCache>();
  statCache->InjectDirectory(".", 41);
  statCache->InjectFile("foo.cpp", 42);
  statCache->InjectFile("bar.cpp", 43);
  manager.setStatCache(std::move(statCache));

  auto fileFoo = manager.getFile("foo.cpp");
  auto fileBar = manager.getFile("bar.cpp");
  ASSERT_TRUE(fileFoo);
  ASSERT_TRUE((*fileFoo)->isValid());
  ASSERT_TRUE(fileBar);
  ASSERT_TRUE((*fileBar)->isValid());
  EXPECT_NE(*fileFoo, *fileBar);
}

// getFile() returns an error if neither a real file nor a virtual file
// exists at the given path.
TEST_F(FileManagerTest, getFileReturnsErrorForNonexistentFile) {
  // Inject a fake foo.cpp into the file system.
  auto statCache = std::make_unique<FakeStatCache>();
  statCache->InjectDirectory(".", 41);
  statCache->InjectFile("foo.cpp", 42);
  statCache->InjectDirectory("MyDirectory", 49);
  manager.setStatCache(std::move(statCache));

  // Create a virtual bar.cpp file.
  manager.getVirtualFile("bar.cpp", 200, 0);

  auto file = manager.getFile("xyz.txt");
  ASSERT_FALSE(file);
  ASSERT_EQ(file.getError(), std::errc::no_such_file_or_directory);

  auto readingDirAsFile = manager.getFile("MyDirectory");
  ASSERT_FALSE(readingDirAsFile);
  ASSERT_EQ(readingDirAsFile.getError(), std::errc::is_a_directory);

  auto readingFileAsDir = manager.getDirectory("foo.cpp");
  ASSERT_FALSE(readingFileAsDir);
  ASSERT_EQ(readingFileAsDir.getError(), std::errc::not_a_directory);
}

// The following tests apply to Unix-like system only.

#ifndef _WIN32

// getFile() returns the same FileEntry for real files that are aliases.
TEST_F(FileManagerTest, getFileReturnsSameFileEntryForAliasedRealFiles) {
  // Inject two real files with the same inode.
  auto statCache = std::make_unique<FakeStatCache>();
  statCache->InjectDirectory("abc", 41);
  statCache->InjectFile("abc/foo.cpp", 42);
  statCache->InjectFile("abc/bar.cpp", 42);
  manager.setStatCache(std::move(statCache));

  auto f1 = manager.getFile("abc/foo.cpp");
  auto f2 = manager.getFile("abc/bar.cpp");

  EXPECT_EQ(f1 ? *f1 : nullptr,
            f2 ? *f2 : nullptr);
}

// getFile() returns the same FileEntry for virtual files that have
// corresponding real files that are aliases.
TEST_F(FileManagerTest, getFileReturnsSameFileEntryForAliasedVirtualFiles) {
  // Inject two real files with the same inode.
  auto statCache = std::make_unique<FakeStatCache>();
  statCache->InjectDirectory("abc", 41);
  statCache->InjectFile("abc/foo.cpp", 42);
  statCache->InjectFile("abc/bar.cpp", 42);
  manager.setStatCache(std::move(statCache));

  ASSERT_TRUE(manager.getVirtualFile("abc/foo.cpp", 100, 0)->isValid());
  ASSERT_TRUE(manager.getVirtualFile("abc/bar.cpp", 200, 0)->isValid());

  auto f1 = manager.getFile("abc/foo.cpp");
  auto f2 = manager.getFile("abc/bar.cpp");

  EXPECT_EQ(f1 ? *f1 : nullptr,
            f2 ? *f2 : nullptr);
}

// getFile() Should return the same entry as getVirtualFile if the file actually
// is a virtual file, even if the name is not exactly the same (but is after
// normalisation done by the file system, like on Windows). This can be checked
// here by checking the size.
TEST_F(FileManagerTest, getVirtualFileWithDifferentName) {
  // Inject fake files into the file system.
  auto statCache = std::make_unique<FakeStatCache>();
  statCache->InjectDirectory("c:\\tmp", 42);
  statCache->InjectFile("c:\\tmp\\test", 43);

  manager.setStatCache(std::move(statCache));

  // Inject the virtual file:
  const FileEntry *file1 = manager.getVirtualFile("c:\\tmp\\test", 123, 1);
  ASSERT_TRUE(file1 != nullptr);
  ASSERT_TRUE(file1->isValid());
  EXPECT_EQ(43U, file1->getUniqueID().getFile());
  EXPECT_EQ(123, file1->getSize());

  // Lookup the virtual file with a different name:
  auto file2 = manager.getFile("c:/tmp/test", 100, 1);
  ASSERT_TRUE(file2);
  ASSERT_TRUE((*file2)->isValid());
  // Check that it's the same UFE:
  EXPECT_EQ(file1, *file2);
  EXPECT_EQ(43U, (*file2)->getUniqueID().getFile());
  // Check that the contents of the UFE are not overwritten by the entry in the
  // filesystem:
  EXPECT_EQ(123, (*file2)->getSize());
}

#endif  // !_WIN32

TEST_F(FileManagerTest, makeAbsoluteUsesVFS) {
  SmallString<64> CustomWorkingDir;
#ifdef _WIN32
  CustomWorkingDir = "C:";
#else
  CustomWorkingDir = "/";
#endif
  llvm::sys::path::append(CustomWorkingDir, "some", "weird", "path");

  auto FS = IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem>(
      new llvm::vfs::InMemoryFileSystem);
  // setCurrentworkingdirectory must finish without error.
  ASSERT_TRUE(!FS->setCurrentWorkingDirectory(CustomWorkingDir));

  FileSystemOptions Opts;
  FileManager Manager(Opts, FS);

  SmallString<64> Path("a/foo.cpp");

  SmallString<64> ExpectedResult(CustomWorkingDir);
  llvm::sys::path::append(ExpectedResult, Path);

  ASSERT_TRUE(Manager.makeAbsolutePath(Path));
  EXPECT_EQ(Path, ExpectedResult);
}

// getVirtualFile should always fill the real path.
TEST_F(FileManagerTest, getVirtualFileFillsRealPathName) {
  SmallString<64> CustomWorkingDir;
#ifdef _WIN32
  CustomWorkingDir = "C:/";
#else
  CustomWorkingDir = "/";
#endif

  auto FS = IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem>(
      new llvm::vfs::InMemoryFileSystem);
  // setCurrentworkingdirectory must finish without error.
  ASSERT_TRUE(!FS->setCurrentWorkingDirectory(CustomWorkingDir));

  FileSystemOptions Opts;
  FileManager Manager(Opts, FS);

  // Inject fake files into the file system.
  auto statCache = std::make_unique<FakeStatCache>();
  statCache->InjectDirectory("/tmp", 42);
  statCache->InjectFile("/tmp/test", 43);

  Manager.setStatCache(std::move(statCache));

  // Check for real path.
  const FileEntry *file = Manager.getVirtualFile("/tmp/test", 123, 1);
  ASSERT_TRUE(file != nullptr);
  ASSERT_TRUE(file->isValid());
  SmallString<64> ExpectedResult = CustomWorkingDir;

  llvm::sys::path::append(ExpectedResult, "tmp", "test");
  EXPECT_EQ(file->tryGetRealPathName(), ExpectedResult);
}

TEST_F(FileManagerTest, getFileDontOpenRealPath) {
  SmallString<64> CustomWorkingDir;
#ifdef _WIN32
  CustomWorkingDir = "C:/";
#else
  CustomWorkingDir = "/";
#endif

  auto FS = IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem>(
      new llvm::vfs::InMemoryFileSystem);
  // setCurrentworkingdirectory must finish without error.
  ASSERT_TRUE(!FS->setCurrentWorkingDirectory(CustomWorkingDir));

  FileSystemOptions Opts;
  FileManager Manager(Opts, FS);

  // Inject fake files into the file system.
  auto statCache = std::make_unique<FakeStatCache>();
  statCache->InjectDirectory("/tmp", 42);
  statCache->InjectFile("/tmp/test", 43);

  Manager.setStatCache(std::move(statCache));

  // Check for real path.
  auto file = Manager.getFile("/tmp/test", /*OpenFile=*/false);
  ASSERT_TRUE(file);
  ASSERT_TRUE((*file)->isValid());
  SmallString<64> ExpectedResult = CustomWorkingDir;

  llvm::sys::path::append(ExpectedResult, "tmp", "test");
  EXPECT_EQ((*file)->tryGetRealPathName(), ExpectedResult);
}

TEST_F(FileManagerTest, getBypassFile) {
  SmallString<64> CustomWorkingDir;
#ifdef _WIN32
  CustomWorkingDir = "C:/";
#else
  CustomWorkingDir = "/";
#endif

  auto FS = IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem>(
      new llvm::vfs::InMemoryFileSystem);
  // setCurrentworkingdirectory must finish without error.
  ASSERT_TRUE(!FS->setCurrentWorkingDirectory(CustomWorkingDir));

  FileSystemOptions Opts;
  FileManager Manager(Opts, FS);

  // Inject fake files into the file system.
  auto Cache = std::make_unique<FakeStatCache>();
  Cache->InjectDirectory("/tmp", 42);
  Cache->InjectFile("/tmp/test", 43);
  Manager.setStatCache(std::move(Cache));

  // Set up a virtual file with a different size than FakeStatCache uses.
  const FileEntry *File = Manager.getVirtualFile("/tmp/test", /*Size=*/10, 0);
  ASSERT_TRUE(File);
  FileEntryRef Ref("/tmp/test", *File);
  EXPECT_TRUE(Ref.isValid());
  EXPECT_EQ(Ref.getSize(), 10);

  // Calling a second time should not affect the UID or size.
  unsigned VirtualUID = Ref.getUID();
  EXPECT_EQ(*expectedToOptional(Manager.getFileRef("/tmp/test")), Ref);
  EXPECT_EQ(Ref.getUID(), VirtualUID);
  EXPECT_EQ(Ref.getSize(), 10);

  // Bypass the file.
  llvm::Optional<FileEntryRef> BypassRef = Manager.getBypassFile(Ref);
  ASSERT_TRUE(BypassRef);
  EXPECT_TRUE(BypassRef->isValid());
  EXPECT_EQ(BypassRef->getName(), Ref.getName());

  // Check that it's different in the right ways.
  EXPECT_NE(&BypassRef->getFileEntry(), File);
  EXPECT_NE(BypassRef->getUID(), VirtualUID);
  EXPECT_NE(BypassRef->getSize(), Ref.getSize());

  // The virtual file should still be returned when searching.
  EXPECT_EQ(*expectedToOptional(Manager.getFileRef("/tmp/test")), Ref);
}

} // anonymous namespace
