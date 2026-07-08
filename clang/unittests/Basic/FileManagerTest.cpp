//===- unittests/Basic/FileMangerTest.cpp ------------ FileManger tests ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

namespace {

// The test fixture.
class FileManagerTest : public ::testing::Test {
protected:
  FileSystemOptions options;
  IntrusiveRefCntPtr<vfs::InMemoryFileSystem> FS =
      llvm::makeIntrusiveRefCnt<vfs::InMemoryFileSystem>();
  FileManager manager{options, FS};

  time_t ModTime = 0;
  std::unique_ptr<MemoryBuffer> EmptyBuf = MemoryBuffer::getMemBuffer("");
};

// When a virtual file is added, its getDir() field has correct name.
TEST_F(FileManagerTest, getVirtualFileSetsTheDirFieldCorrectly) {
  FileEntryRef file = manager.getVirtualFileRef("foo.cpp", 42, 0);
  EXPECT_EQ(".", file.getDir().getName());

  file = manager.getVirtualFileRef("x/y/z.cpp", 42, 0);
  EXPECT_EQ("x/y", file.getDir().getName());
}

// Before any virtual file is added, no virtual directory exists.
TEST_F(FileManagerTest, NoVirtualDirectoryExistsBeforeAVirtualFileIsAdded) {
  ASSERT_FALSE(manager.getOptionalDirectoryRef("virtual/dir/foo"));
  ASSERT_FALSE(manager.getOptionalDirectoryRef("virtual/dir"));
  ASSERT_FALSE(manager.getOptionalDirectoryRef("virtual"));
}

// When a virtual file is added, all of its ancestors should be created.
TEST_F(FileManagerTest, getVirtualFileCreatesDirectoryEntriesForAncestors) {
  manager.getVirtualFileRef("virtual/dir/bar.h", 100, 0);

  auto dir = manager.getDirectoryRef("virtual/dir/foo");
  ASSERT_THAT_EXPECTED(dir, llvm::Failed());

  dir = manager.getDirectoryRef("virtual/dir");
  ASSERT_THAT_EXPECTED(dir, llvm::Succeeded());
  EXPECT_EQ("virtual/dir", dir->getName());

  dir = manager.getDirectoryRef("virtual");
  ASSERT_THAT_EXPECTED(dir, llvm::Succeeded());
  EXPECT_EQ("virtual", dir->getName());
}

// getFileRef() succeeds if a real file exists at the given path.
TEST_F(FileManagerTest, getFileReturnsValidFileEntryForExistingRealFile) {
  // Inject fake files into the file system.
  FS->addFileNoOwn("/tmp/test", ModTime, *EmptyBuf);

  auto file = manager.getFileRef("/tmp/test");
  ASSERT_THAT_EXPECTED(file, llvm::Succeeded());
  EXPECT_EQ("/tmp/test", file->getName());
  EXPECT_EQ("/tmp", file->getDir().getName());
}

// getFileRef() succeeds if a virtual file exists at the given path.
TEST_F(FileManagerTest, getFileReturnsValidFileEntryForExistingVirtualFile) {
  manager.getVirtualFileRef("virtual/dir/bar.h", 100, 0);
  auto file = manager.getFileRef("virtual/dir/bar.h");
  ASSERT_THAT_EXPECTED(file, llvm::Succeeded());
  EXPECT_EQ("virtual/dir/bar.h", file->getName());
  EXPECT_EQ("virtual/dir", file->getDir().getName());
}

// getFile() returns different FileEntries for different paths when
// there's no aliasing.
TEST_F(FileManagerTest, getFileReturnsDifferentFileEntriesForDifferentFiles) {
  // Inject two fake files into the file system.
  FS->addFileNoOwn("foo.cpp", ModTime, *EmptyBuf);
  FS->addFileNoOwn("bar.cpp", ModTime, *EmptyBuf);

  auto fileFoo = manager.getOptionalFileRef("foo.cpp");
  auto fileBar = manager.getOptionalFileRef("bar.cpp");
  ASSERT_TRUE(fileFoo);
  ASSERT_TRUE(fileBar);
  EXPECT_NE(&fileFoo->getFileEntry(), &fileBar->getFileEntry());
}

// getFile() returns an error if neither a real file nor a virtual file
// exists at the given path.
TEST_F(FileManagerTest, getFileReturnsErrorForNonexistentFile) {
  // Inject a fake foo.cpp into the file system.
  FS->addFileNoOwn("foo.cpp", ModTime, *EmptyBuf);
  FS->addFileNoOwn("MyDirectory/keep", ModTime, *EmptyBuf);

  // Create a virtual bar.cpp file.
  manager.getVirtualFileRef("bar.cpp", 200, 0);

  auto file = manager.getFileRef("xyz.txt");
  ASSERT_FALSE(file);
  ASSERT_EQ(llvm::errorToErrorCode(file.takeError()),
            std::make_error_code(std::errc::no_such_file_or_directory));

  auto readingDirAsFile = manager.getFileRef("MyDirectory");
  ASSERT_FALSE(readingDirAsFile);
  ASSERT_EQ(llvm::errorToErrorCode(readingDirAsFile.takeError()),
            std::make_error_code(std::errc::is_a_directory));

  auto readingFileAsDir = manager.getDirectoryRef("foo.cpp");
  ASSERT_FALSE(readingFileAsDir);
  ASSERT_EQ(llvm::errorToErrorCode(readingFileAsDir.takeError()),
            std::make_error_code(std::errc::not_a_directory));
}

TEST_F(FileManagerTest, getFileRefErrorIncludesFilename) {
  auto FS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  auto EmptyBuffer = llvm::MemoryBuffer::getMemBuffer("");
  ASSERT_TRUE(
      FS->addFileNoOwn("/MyDirectory/file", 0, EmptyBuffer->getMemBufferRef()));
  FileSystemOptions Opts;
  FileManager Mgr(Opts, FS);

  // Build the expected message for a given filename and error code, since the
  // system-provided message text (and capitalization) for std::errc values
  // varies by platform.
  auto ExpectedMsg = [](StringRef Name, std::errc EC) {
    return ("'" + Name + "': " + std::make_error_code(EC).message()).str();
  };

  // Nonexistent file.
  auto Missing = Mgr.getFileRef("/xyz.txt");
  ASSERT_FALSE(Missing);
  EXPECT_EQ(ExpectedMsg("/xyz.txt", std::errc::no_such_file_or_directory),
            llvm::toString(Missing.takeError()));

  // Cached failure
  auto MissingAgain = Mgr.getFileRef("/xyz.txt");
  ASSERT_FALSE(MissingAgain);
  EXPECT_EQ(std::make_error_code(std::errc::no_such_file_or_directory),
            llvm::errorToErrorCode(MissingAgain.takeError()));

  // Reading a directory as a file.
  auto DirAsFile = Mgr.getFileRef("/MyDirectory");
  ASSERT_FALSE(DirAsFile);
  EXPECT_EQ(ExpectedMsg("/MyDirectory", std::errc::is_a_directory),
            llvm::toString(DirAsFile.takeError()));

  auto Trailing = Mgr.getFileRef("/some/dir/");
  ASSERT_FALSE(Trailing);
  EXPECT_EQ(ExpectedMsg("/some/dir/", std::errc::is_a_directory),
            llvm::toString(Trailing.takeError()));
}

TEST_F(FileManagerTest, getDirectoryRefErrorIncludesFilename) {
  auto FS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  auto EmptyBuffer = llvm::MemoryBuffer::getMemBuffer("");
  ASSERT_TRUE(FS->addFileNoOwn("/foo.cpp", 0, EmptyBuffer->getMemBufferRef()));
  FileSystemOptions Opts;
  FileManager Mgr(Opts, FS);

  auto ExpectedMsg = [](StringRef Name, std::errc EC) {
    return ("'" + Name + "': " + std::make_error_code(EC).message()).str();
  };

  // Nonexistent directory.
  auto Missing = Mgr.getDirectoryRef("/no_such_dir");
  ASSERT_FALSE(Missing);
  EXPECT_EQ(ExpectedMsg("/no_such_dir", std::errc::no_such_file_or_directory),
            llvm::toString(Missing.takeError()));

  // Cached failure
  auto MissingAgain = Mgr.getDirectoryRef("/no_such_dir");
  ASSERT_FALSE(MissingAgain);
  EXPECT_EQ(std::make_error_code(std::errc::no_such_file_or_directory),
            llvm::errorToErrorCode(MissingAgain.takeError()));

  // Reading a file as a directory.
  auto FileAsDir = Mgr.getDirectoryRef("/foo.cpp");
  ASSERT_FALSE(FileAsDir);
  EXPECT_EQ(ExpectedMsg("/foo.cpp", std::errc::not_a_directory),
            llvm::toString(FileAsDir.takeError()));
}

// getFile() returns the same FileEntry for real files that are aliases.
TEST_F(FileManagerTest, getFileReturnsSameFileEntryForAliasedRealFiles) {
  // Inject two real files with the same inode.
  FS->addFileNoOwn("abc/foo.cpp", ModTime, *EmptyBuf);
  FS->addHardLink("abc/bar.cpp", "abc/foo.cpp");

  auto f1 = manager.getOptionalFileRef("abc/foo.cpp");
  auto f2 = manager.getOptionalFileRef("abc/bar.cpp");

  EXPECT_EQ(f1 ? &f1->getFileEntry() : nullptr,
            f2 ? &f2->getFileEntry() : nullptr);

  // Check that getFileRef also does the right thing.
  auto r1 = manager.getFileRef("abc/foo.cpp");
  auto r2 = manager.getFileRef("abc/bar.cpp");
  ASSERT_FALSE(!r1);
  ASSERT_FALSE(!r2);

  EXPECT_EQ("abc/foo.cpp", r1->getName());
  EXPECT_EQ("abc/bar.cpp", r2->getName());
  EXPECT_EQ((f1 ? &f1->getFileEntry() : nullptr), &r1->getFileEntry());
  EXPECT_EQ((f2 ? &f2->getFileEntry() : nullptr), &r2->getFileEntry());
}

TEST_F(FileManagerTest, getFileRefReturnsCorrectNameForDifferentStatPath) {
  // This is adding coverage for stat behaviour triggered by the
  // RedirectingFileSystem for 'use-external-name' that FileManager::getFileRef
  // has special logic for.
  FS->addFileNoOwn("/dir/f1.cpp", ModTime, *EmptyBuf);
  FS->addFileNoOwn("/dir/f2.cpp", ModTime, *EmptyBuf);

  // This unintuitive rename-the-file-on-stat behaviour supports how the
  // RedirectingFileSystem VFS layer responds to stats. However, even if you
  // have two layers, you should only get a single filename back. As such the
  // following stat cache behaviour is not supported (the correct stat entry
  // for a double-redirection would be "dir/f1.cpp") and the getFileRef below
  // should assert.
  IntrusiveRefCntPtr<vfs::FileSystem> NewFS = FS;

  NewFS = vfs::RedirectingFileSystem::create(
      {{"/dir/f1-alias.cpp", "/dir/f1.cpp"},
       {"/dir/f2-alias.cpp", "/dir/f2.cpp"}},
      /*UseExternalNames=*/true, std::move(NewFS));

  NewFS = vfs::RedirectingFileSystem::create(
      {{"/dir/f1-alias-alias.cpp", "/dir/f1-alias.cpp"}},
      /*UseExternalNames=*/true, std::move(NewFS));

  manager.setVirtualFileSystem(std::move(NewFS));

  // With F1, test accessing the non-redirected name first.
  auto F1 = manager.getFileRef("/dir/f1.cpp");
  auto F1Alias = manager.getFileRef("/dir/f1-alias.cpp");
  auto F1Alias2 = manager.getFileRef("/dir/f1-alias.cpp");
  ASSERT_FALSE(!F1);
  ASSERT_FALSE(!F1Alias);
  ASSERT_FALSE(!F1Alias2);
  EXPECT_EQ("/dir/f1.cpp", F1->getName());
  EXPECT_EQ("/dir/f1.cpp", F1Alias->getName());
  EXPECT_EQ("/dir/f1.cpp", F1Alias2->getName());
  EXPECT_EQ(&F1->getFileEntry(), &F1Alias->getFileEntry());
  EXPECT_EQ(&F1->getFileEntry(), &F1Alias2->getFileEntry());

#if !defined(NDEBUG) && GTEST_HAS_DEATH_TEST
  EXPECT_DEATH((void)manager.getOptionalFileRef("/dir/f1-alias-alias.cpp"),
               "filename redirected to a non-canonical filename?");
#endif

  // With F2, test accessing the redirected name first.
  auto F2Alias = manager.getFileRef("/dir/f2-alias.cpp");
  auto F2 = manager.getFileRef("/dir/f2.cpp");
  auto F2Alias2 = manager.getFileRef("/dir/f2-alias.cpp");
  ASSERT_FALSE(!F2);
  ASSERT_FALSE(!F2Alias);
  ASSERT_FALSE(!F2Alias2);
  EXPECT_EQ("/dir/f2.cpp", F2->getName());
  EXPECT_EQ("/dir/f2.cpp", F2Alias->getName());
  EXPECT_EQ("/dir/f2.cpp", F2Alias2->getName());
  EXPECT_EQ(&F2->getFileEntry(), &F2Alias->getFileEntry());
  EXPECT_EQ(&F2->getFileEntry(), &F2Alias2->getFileEntry());
}

TEST_F(FileManagerTest, getFileRefReturnsCorrectDirNameForDifferentStatPath) {
  // Inject files with the same inode into distinct directories (name & inode).
  FS->addFileNoOwn("dir1/f.cpp", ModTime, *EmptyBuf);
  FS->addHardLink("dir2/f.cpp", "dir1/f.cpp");

  auto Dir1F = manager.getFileRef("dir1/f.cpp");
  auto Dir2F = manager.getFileRef("dir2/f.cpp");

  ASSERT_FALSE(!Dir1F);
  ASSERT_FALSE(!Dir2F);
  EXPECT_EQ("dir1", Dir1F->getDir().getName());
  EXPECT_EQ("dir2", Dir2F->getDir().getName());
  EXPECT_EQ("dir1/f.cpp", Dir1F->getNameAsRequested());
  EXPECT_EQ("dir2/f.cpp", Dir2F->getNameAsRequested());
}

// getFile() returns the same FileEntry for virtual files that have
// corresponding real files that are aliases.
TEST_F(FileManagerTest, getFileReturnsSameFileEntryForAliasedVirtualFiles) {
  // Inject two real files with the same inode.
  FS->addFileNoOwn("abc/foo.cpp", ModTime, *EmptyBuf);
  FS->addHardLink("abc/bar.cpp", "abc/foo.cpp");

  auto f1 = manager.getOptionalFileRef("abc/foo.cpp");
  auto f2 = manager.getOptionalFileRef("abc/bar.cpp");

  EXPECT_EQ(f1 ? &f1->getFileEntry() : nullptr,
            f2 ? &f2->getFileEntry() : nullptr);
}

TEST_F(FileManagerTest, getFileRefEquality) {
  FS->addFileNoOwn("/dir/f1.cpp", ModTime, *EmptyBuf);
  FS->addHardLink("/dir/f1-also.cpp", "/dir/f1.cpp");
  FS->addFileNoOwn("/dir/f2.cpp", ModTime, *EmptyBuf);

  IntrusiveRefCntPtr<vfs::FileSystem> NewFS =
      vfs::RedirectingFileSystem::create(
          {{"/dir/f1-redirect.cpp", "/dir/f1.cpp"}},
          /*UseExternalNames=*/true, FS);

  manager.setVirtualFileSystem(std::move(NewFS));

  auto F1 = manager.getFileRef("/dir/f1.cpp");
  auto F1Again = manager.getFileRef("/dir/f1.cpp");
  auto F1Also = manager.getFileRef("/dir/f1-also.cpp");
  auto F1Redirect = manager.getFileRef("/dir/f1-redirect.cpp");
  auto F1RedirectAgain = manager.getFileRef("/dir/f1-redirect.cpp");
  auto F2 = manager.getFileRef("/dir/f2.cpp");

  // Check Expected<FileEntryRef> for error.
  ASSERT_FALSE(!F1);
  ASSERT_FALSE(!F1Also);
  ASSERT_FALSE(!F1Again);
  ASSERT_FALSE(!F1Redirect);
  ASSERT_FALSE(!F1RedirectAgain);
  ASSERT_FALSE(!F2);

  // Check names.
  EXPECT_EQ("/dir/f1.cpp", F1->getName());
  EXPECT_EQ("/dir/f1.cpp", F1Again->getName());
  EXPECT_EQ("/dir/f1-also.cpp", F1Also->getName());
  EXPECT_EQ("/dir/f1.cpp", F1Redirect->getName());
  EXPECT_EQ("/dir/f1.cpp", F1RedirectAgain->getName());
  EXPECT_EQ("/dir/f2.cpp", F2->getName());

  EXPECT_EQ("/dir/f1.cpp", F1->getNameAsRequested());
  EXPECT_EQ("/dir/f1-redirect.cpp", F1Redirect->getNameAsRequested());

  // Compare against FileEntry*.
  EXPECT_EQ(&F1->getFileEntry(), *F1);
  EXPECT_EQ(*F1, &F1->getFileEntry());
  EXPECT_EQ(&F1->getFileEntry(), &F1Redirect->getFileEntry());
  EXPECT_EQ(&F1->getFileEntry(), &F1RedirectAgain->getFileEntry());
  EXPECT_NE(&F2->getFileEntry(), *F1);
  EXPECT_NE(*F1, &F2->getFileEntry());

  // Compare using ==.
  EXPECT_EQ(*F1, *F1Also);
  EXPECT_EQ(*F1, *F1Again);
  EXPECT_EQ(*F1, *F1Redirect);
  EXPECT_EQ(*F1Also, *F1Redirect);
  EXPECT_EQ(*F1, *F1RedirectAgain);
  EXPECT_NE(*F2, *F1);
  EXPECT_NE(*F2, *F1Also);
  EXPECT_NE(*F2, *F1Again);
  EXPECT_NE(*F2, *F1Redirect);

  // Compare using isSameRef.
  EXPECT_TRUE(F1->isSameRef(*F1Again));
  EXPECT_FALSE(F1->isSameRef(*F1Redirect));
  EXPECT_FALSE(F1->isSameRef(*F1Also));
  EXPECT_FALSE(F1->isSameRef(*F2));
  EXPECT_TRUE(F1Redirect->isSameRef(*F1RedirectAgain));
}

// getFile() Should return the same entry as getVirtualFile if the file actually
// is a virtual file, even if the name is not exactly the same (but is after
// normalisation done by the file system, like on Windows). This can be checked
// here by checking the size.
TEST_F(FileManagerTest, getVirtualFileWithDifferentName) {
  // Inject fake files into the file system.
  FS->addFileNoOwn("/tmp/test", ModTime, *EmptyBuf);

  // Inject the virtual file:
  FileEntryRef file1 = manager.getVirtualFileRef("/tmp/test", 123, 1);
  EXPECT_EQ(123, file1.getSize());

  // Lookup the virtual file with a different name:
  auto file2 = manager.getOptionalFileRef("/tmp/./test", 100, 1);
  ASSERT_TRUE(file2);
  // Check that it's the same UFE:
  EXPECT_EQ(file1, *file2);
  // Check that the contents of the UFE are not overwritten by the entry in the
  // filesystem:
  EXPECT_EQ(123, file2->getSize());
}

static StringRef getSystemRoot() {
  return is_style_windows(llvm::sys::path::Style::native) ? "C:\\" : "/";
}

TEST_F(FileManagerTest, makeAbsoluteUsesVFS) {
  // FIXME: Should this be using a root path / call getSystemRoot()? For now,
  // avoiding that and leaving the test as-is.
  SmallString<64> CustomWorkingDir =
      is_style_windows(llvm::sys::path::Style::native) ? StringRef("C:")
                                                       : StringRef("/");
  llvm::sys::path::append(CustomWorkingDir, "some", "weird", "path");

  // setCurrentWorkingDirectory must finish without error.
  ASSERT_TRUE(!FS->setCurrentWorkingDirectory(CustomWorkingDir));

  SmallString<64> Path("a/foo.cpp");

  SmallString<64> ExpectedResult(CustomWorkingDir);
  llvm::sys::path::append(ExpectedResult, Path);

  ASSERT_TRUE(manager.makeAbsolutePath(Path));
  EXPECT_EQ(Path, ExpectedResult);
}

// getVirtualFile should always fill the real path.
TEST_F(FileManagerTest, getVirtualFileFillsRealPathName) {
  SmallString<64> CustomWorkingDir = getSystemRoot();

  // setCurrentWorkingDirectory must finish without error.
  ASSERT_TRUE(!FS->setCurrentWorkingDirectory(CustomWorkingDir));

  // Inject fake files into the file system.
  FS->addFileNoOwn("/tmp/test", ModTime, *EmptyBuf);

  // Check for real path.
  FileEntryRef file = manager.getVirtualFileRef("/tmp/test", 123, 1);
  SmallString<64> ExpectedResult = CustomWorkingDir;

  llvm::sys::path::append(ExpectedResult, "tmp", "test");
  // Normalize to native path style to match tryGetRealPathName()
  // which uses native style (potentially forward slashes on Windows
  // if LLVM_WINDOWS_PREFER_FORWARD_SLASH is on).
  llvm::sys::path::native(ExpectedResult);
  EXPECT_EQ(file.getFileEntry().tryGetRealPathName(), ExpectedResult);
}

TEST_F(FileManagerTest, getFileDontOpenRealPath) {
  SmallString<64> CustomWorkingDir = getSystemRoot();

  // setCurrentWorkingDirectory must finish without error.
  ASSERT_TRUE(!FS->setCurrentWorkingDirectory(CustomWorkingDir));

  // Inject fake files into the file system.
  FS->addFileNoOwn("/tmp/test", ModTime, *EmptyBuf);

  // Check for real path.
  auto file = manager.getOptionalFileRef("/tmp/test", /*OpenFile=*/false);
  ASSERT_TRUE(file);
  SmallString<64> ExpectedResult = CustomWorkingDir;

  llvm::sys::path::append(ExpectedResult, "tmp", "test");
  // Normalize to native path style to match tryGetRealPathName()
  // which uses native style (potentially forward slashes on Windows
  // if LLVM_WINDOWS_PREFER_FORWARD_SLASH is on).
  llvm::sys::path::native(ExpectedResult);
  EXPECT_EQ(file->getFileEntry().tryGetRealPathName(), ExpectedResult);
}

TEST_F(FileManagerTest, getBypassFile) {
  SmallString<64> CustomWorkingDir;
#ifdef _WIN32
  CustomWorkingDir = "C:/";
#else
  CustomWorkingDir = "/";
#endif

  // setCurrentWorkingDirectory must finish without error.
  ASSERT_TRUE(!FS->setCurrentWorkingDirectory(CustomWorkingDir));

  // Inject fake files into the file system.
  FS->addFileNoOwn("/tmp/test", ModTime, *EmptyBuf);

  // Set up a virtual file with a different size than the VFS uses.
  FileEntryRef File = manager.getVirtualFileRef("/tmp/test", /*Size=*/10, 0);
  ASSERT_TRUE(File);
  const FileEntry &FE = *File;
  EXPECT_EQ(FE.getSize(), 10);

  // Calling a second time should not affect the UID or size.
  unsigned VirtualUID = FE.getUID();
  OptionalFileEntryRef SearchRef;
  ASSERT_THAT_ERROR(manager.getFileRef("/tmp/test").moveInto(SearchRef),
                    Succeeded());
  EXPECT_EQ(&FE, &SearchRef->getFileEntry());
  EXPECT_EQ(FE.getUID(), VirtualUID);
  EXPECT_EQ(FE.getSize(), 10);

  // Bypass the file.
  OptionalFileEntryRef BypassRef = manager.getBypassFile(File);
  ASSERT_TRUE(BypassRef);
  EXPECT_EQ("/tmp/test", BypassRef->getName());

  // Check that it's different in the right ways.
  EXPECT_NE(&BypassRef->getFileEntry(), File);
  EXPECT_NE(BypassRef->getUID(), VirtualUID);
  EXPECT_NE(BypassRef->getSize(), FE.getSize());

  // The virtual file should still be returned when searching.
  ASSERT_THAT_ERROR(manager.getFileRef("/tmp/test").moveInto(SearchRef),
                    Succeeded());
  EXPECT_EQ(&FE, &SearchRef->getFileEntry());
}

} // anonymous namespace
