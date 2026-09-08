//===-- DraftStoreTests.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DraftStore.h"
#include "FS.h"
#include "TestFS.h"
#include "clang/Basic/FileManager.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

using ::testing::UnorderedElementsAre;

std::vector<std::string> directoryEntries(llvm::vfs::FileSystem &FS,
                                          llvm::StringRef Directory) {
  std::vector<std::string> Result;
  std::error_code EC;
  for (auto It = FS.dir_begin(Directory, EC),
            End = llvm::vfs::directory_iterator();
       !EC && It != End; It.increment(EC)) {
    auto Style = It->path().contains('\\') ? llvm::sys::path::Style::windows
                                           : llvm::sys::path::Style::native;
    Result.push_back(llvm::sys::path::filename(It->path(), Style).str());
  }
  EXPECT_FALSE(EC) << EC.message();
  return Result;
}

TEST(DraftStore, Versions) {
  DraftStore DS;
  Path File = "foo.cpp";

  EXPECT_EQ("25", DS.addDraft(File, "25", ""));
  EXPECT_EQ("25", DS.getDraft(File)->Version);
  EXPECT_EQ("", *DS.getDraft(File)->Contents);

  EXPECT_EQ("26", DS.addDraft(File, "", "x"));
  EXPECT_EQ("26", DS.getDraft(File)->Version);
  EXPECT_EQ("x", *DS.getDraft(File)->Contents);

  EXPECT_EQ("27", DS.addDraft(File, "", "x")) << "no-op change";
  EXPECT_EQ("27", DS.getDraft(File)->Version);
  EXPECT_EQ("x", *DS.getDraft(File)->Contents);

  // We allow versions to go backwards.
  EXPECT_EQ("7", DS.addDraft(File, "7", "y"));
  EXPECT_EQ("7", DS.getDraft(File)->Version);
  EXPECT_EQ("y", *DS.getDraft(File)->Contents);
}

TEST(DraftStore, DriveLetterIdentity) {
  DraftStore DS;
  EXPECT_EQ("1", DS.addDraft("C:/proj/a.cpp", "1", "int x;"));
  auto Draft = DS.getDraft("c:/proj/a.cpp");
  ASSERT_TRUE(Draft);
  EXPECT_EQ("int x;", *Draft->Contents);
  EXPECT_EQ("1", Draft->Version);
  DS.removeDraft("c:/proj/a.cpp");
  EXPECT_FALSE(DS.getDraft("C:/proj/a.cpp"));
}

TEST(DraftStore, DistinctFilenameCase) {
  DraftStore DS;
  DS.addDraft("C:/proj/Foo.h", "1", "upper");
  DS.addDraft("C:/proj/foo.h", "2", "lower");
  EXPECT_EQ(DS.getActiveFiles().size(), 2u);
  ASSERT_TRUE(DS.getDraft("c:/proj/Foo.h"));
  ASSERT_TRUE(DS.getDraft("c:/proj/foo.h"));
  EXPECT_EQ(*DS.getDraft("c:/proj/Foo.h")->Contents, "upper");
  EXPECT_EQ(*DS.getDraft("c:/proj/foo.h")->Contents, "lower");

  auto FS = DS.asVFS();
  auto Upper = FS->getBufferForFile("c:/proj/Foo.h");
  auto Lower = FS->getBufferForFile("c:/proj/foo.h");
  ASSERT_TRUE(Upper);
  ASSERT_TRUE(Lower);
  EXPECT_EQ((*Upper)->getBuffer(), "upper");
  EXPECT_EQ((*Lower)->getBuffer(), "lower");
  DS.removeDraft("c:/proj/foo.h");
  EXPECT_TRUE(DS.getDraft("C:/proj/Foo.h"));
  EXPECT_FALSE(DS.getDraft("C:/proj/foo.h"));
}

TEST(DraftStore, AsVFSDriveLetterAndSlash) {
  DraftStore DS;
  DS.addDraft("c:/proj/a.cpp", "1", "int unsaved;");
  auto FS = DS.asVFS();

  for (const char *P : {"C:/proj/a.cpp", "c:/proj/a.cpp", "C:\\proj\\a.cpp",
                        "c:\\proj\\a.cpp"}) {
    auto S = FS->status(P);
    ASSERT_TRUE(S) << P;
    EXPECT_TRUE(S->isRegularFile());
    auto Buf = FS->getBufferForFile(P);
    ASSERT_TRUE(Buf) << P;
    EXPECT_EQ((*Buf)->getBuffer(), "int unsaved;");
  }
  EXPECT_FALSE(FS->status("C:/proj/missing.cpp"));
  EXPECT_FALSE(FS->status("D:/proj/a.cpp"));
}

TEST(DraftStore, VFSBufferOwnsContents) {
  std::unique_ptr<llvm::MemoryBuffer> Buffer;
  {
    DraftStore DS;
    DS.addDraft("/proj/a.cpp", "1", "old contents");
    auto FS = DS.asVFS();
    auto Result = FS->getBufferForFile("/proj/a.cpp");
    ASSERT_TRUE(Result);
    Buffer = std::move(*Result);
  }
  EXPECT_EQ(Buffer->getBuffer(), "old contents");
}

TEST(DraftStore, VFSDirectoriesAndMetadata) {
  DraftStore DS;
  DS.addDraft("c:/proj/src/a.cpp", "1", "int unsaved;");
  auto FS = DS.asVFS();

  auto Dir = FS->status("C:\\proj\\src");
  ASSERT_TRUE(Dir);
  EXPECT_TRUE(Dir->isDirectory());
  EXPECT_THAT(directoryEntries(*FS, "C:\\proj\\src"),
              UnorderedElementsAre("a.cpp"));

  llvm::SmallString<64> RealPath;
  EXPECT_FALSE(FS->getRealPath("C:\\proj\\src\\a.cpp", RealPath));
  EXPECT_EQ(llvm::sys::path::convert_to_slash(RealPath,
                                              llvm::sys::path::Style::windows),
            "c:/proj/src/a.cpp");
  bool IsLocal = true;
  EXPECT_FALSE(FS->isLocal("C:/proj/src/a.cpp", IsLocal));
  EXPECT_FALSE(IsLocal);
}

TEST(DraftStore, VFSUniqueIDs) {
  DraftStore DS;
  DS.addDraft("C:/proj/a.cpp", "1", "a");
  DS.addDraft("C:/proj/b.cpp", "1", "b");
  auto FS = DS.asVFS();

  auto A = FS->status("C:/proj/a.cpp");
  auto Alias = FS->status("c:\\proj\\a.cpp");
  auto B = FS->status("C:/proj/b.cpp");
  auto Dir = FS->status("C:/proj");
  ASSERT_TRUE(A);
  ASSERT_TRUE(Alias);
  ASSERT_TRUE(B);
  ASSERT_TRUE(Dir);
  EXPECT_EQ(A->getUniqueID(), Alias->getUniqueID());
  EXPECT_NE(A->getUniqueID(), B->getUniqueID());
  EXPECT_NE(A->getUniqueID(), Dir->getUniqueID());
  EXPECT_NE(B->getUniqueID(), Dir->getUniqueID());
}

TEST(DraftStore, VFSUniqueIDsAcrossSnapshots) {
  DraftStore DS;
  auto Original = testPath("original.h");
  DS.addDraft(Original, "1", "struct Original {};");
  auto Before = DS.asVFS();
  auto Status = Before->status(Original);
  ASSERT_TRUE(Status);
  PreambleFileStatusCache Cache(testPath("main.cc"));
  Cache.update(*Before, *Status, Original);

  // A preamble's cached status can outlive the draft snapshot it came from.
  for (int I = 0; I != 16; ++I) {
    auto Added = testPath("new" + std::to_string(I) + ".h");
    DS.addDraft(Added, "1", "struct Added {};");
    auto After = DS.asVFS();
    auto AddedStatus = After->status(Added);
    ASSERT_TRUE(AddedStatus);
    EXPECT_NE(Status->getUniqueID(), AddedStatus->getUniqueID());
    FileManager FM({}, Cache.getConsumingFS(After));
    auto A = FM.getOptionalFileRef(Original);
    auto B = FM.getOptionalFileRef(Added, /*OpenFile=*/true);
    ASSERT_TRUE(A);
    ASSERT_TRUE(B);
    EXPECT_NE(&A->getFileEntry(), &B->getFileEntry());
    DS.removeDraft(Added);
  }
}

TEST(DraftStore, VFSStableIDsWithPreambleCache) {
  DraftStore DS;
  auto Original = testPath("original.h");
  auto Alias = testPath("./original.h");
  DS.addDraft(Original, "1", "#pragma once\nstruct Original {};");
  auto Before = DS.asVFS();
  auto Status = Before->status(Original);
  ASSERT_TRUE(Status);
  PreambleFileStatusCache Cache(testPath("main.cc"));
  Cache.update(*Before, *Status, Original);

  for (bool AddOtherDraft : {false, true}) {
    SCOPED_TRACE(AddOtherDraft);
    if (AddOtherDraft)
      DS.addDraft(testPath("other.h"), "1", "struct Other {};");
    auto After = DS.asVFS();
    auto CurrentStatus = After->status(Original);
    ASSERT_TRUE(CurrentStatus);
    EXPECT_EQ(Status->getUniqueID(), CurrentStatus->getUniqueID());

    auto FS = Cache.getConsumingFS(After);
    auto CachedStatus = FS->status(Original);
    auto Opened = FS->openFileForRead(Alias);
    ASSERT_TRUE(CachedStatus);
    ASSERT_TRUE(Opened);
    auto OpenStatus = (*Opened)->status();
    ASSERT_TRUE(OpenStatus);
    EXPECT_EQ(CachedStatus->getUniqueID(), OpenStatus->getUniqueID());

    // ASTReader uses cached status, while a later include may open an alias.
    FileManager FM({}, FS);
    auto A = FM.getOptionalFileRef(Original);
    auto B = FM.getOptionalFileRef(Alias, /*OpenFile=*/true);
    ASSERT_TRUE(A);
    ASSERT_TRUE(B);
    EXPECT_EQ(&A->getFileEntry(), &B->getFileEntry());
  }
}

TEST(DraftStore, VFSIdentityLifetime) {
  DraftStore DS;
  auto File = testPath("header.h");
  DS.addDraft(File, "1", "old");
  auto Before = DS.asVFS();
  auto Original = Before->status(File);
  ASSERT_TRUE(Original);

  DS.addDraft(File, "2", "new");
  auto After = DS.asVFS();
  auto Updated = After->status(File);
  ASSERT_TRUE(Updated);
  EXPECT_EQ(Original->getUniqueID(), Updated->getUniqueID());
  auto OldBuffer = Before->getBufferForFile(File);
  auto NewBuffer = After->getBufferForFile(File);
  ASSERT_TRUE(OldBuffer);
  ASSERT_TRUE(NewBuffer);
  EXPECT_EQ((*OldBuffer)->getBuffer(), "old");
  EXPECT_EQ((*NewBuffer)->getBuffer(), "new");

  DS.removeDraft(File);
  DS.addDraft(File, "3", "reopened");
  auto Reopened = DS.asVFS()->status(File);
  ASSERT_TRUE(Reopened);
  EXPECT_NE(Original->getUniqueID(), Reopened->getUniqueID());
}

TEST(DraftStore, VFSDottedPaths) {
  for (const char *Name : {"C:/proj/./dirty.h", "C:/proj/sub/../dirty.h",
                           "C:\\proj\\sub\\..\\dirty.h"}) {
    SCOPED_TRACE(Name);
    DraftStore DS;
    DS.addDraft(Name, "1", "unsaved");
    auto FS = DS.asVFS();
    for (const char *Alias : {Name, "C:/proj/dirty.h", "c:\\proj\\dirty.h"}) {
      auto Buffer = FS->getBufferForFile(Alias);
      ASSERT_TRUE(Buffer) << Alias;
      EXPECT_EQ((*Buffer)->getBuffer(), "unsaved");
    }
    EXPECT_THAT(directoryEntries(*FS, "C:/proj"),
                UnorderedElementsAre("dirty.h"));
  }
}

TEST(DraftStore, VFSRelativeWorkingDirectory) {
  for (const auto &Root : {testPath("proj"), std::string("C:/proj")}) {
    SCOPED_TRACE(Root);
    DraftStore DS;
    DS.addDraft(Root + "/src/header.h", "1", "unsaved");
    auto FS = DS.asVFS();
    ASSERT_FALSE(FS->setCurrentWorkingDirectory(Root));

    for (const char *Next : {"src", ".", "", "../src", "nested/.."}) {
      SCOPED_TRACE(Next);
      ASSERT_FALSE(FS->setCurrentWorkingDirectory(Next));
      auto CWD = FS->getCurrentWorkingDirectory();
      ASSERT_TRUE(CWD);
      EXPECT_EQ(PathRef(*CWD), PathRef(Root + "/src"));
      EXPECT_TRUE(FS->status("header.h"));
      auto Buffer = FS->getBufferForFile("header.h");
      ASSERT_TRUE(Buffer);
      EXPECT_EQ((*Buffer)->getBuffer(), "unsaved");
      EXPECT_THAT(directoryEntries(*FS, "."), UnorderedElementsAre("header.h"));
    }

    ASSERT_FALSE(FS->setCurrentWorkingDirectory(".."));
    auto CWD = FS->getCurrentWorkingDirectory();
    ASSERT_TRUE(CWD);
    EXPECT_EQ(PathRef(*CWD), PathRef(Root));
    EXPECT_TRUE(FS->status("src/header.h"));

    ASSERT_FALSE(FS->setCurrentWorkingDirectory(Root + "/src/../src"));
    CWD = FS->getCurrentWorkingDirectory();
    ASSERT_TRUE(CWD);
    EXPECT_EQ(PathRef(*CWD), PathRef(Root + "/src"));
    EXPECT_TRUE(FS->status("header.h"));
  }
}

TEST(DraftStore, VFSRelativeWorkingDirectoryPreservesOverlayPrecedence) {
  auto Header = testPath("proj/src/header.h");
  auto Base = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  ASSERT_TRUE(
      Base->addFile(Header, 0, llvm::MemoryBuffer::getMemBuffer("stale disk")));
  DraftStore DS;
  DS.addDraft(Header, "1", "unsaved");
  auto Overlay = llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(Base);
  Overlay->pushOverlay(DS.asVFS());

  ASSERT_FALSE(Overlay->setCurrentWorkingDirectory(testPath("proj")));
  for (const char *Next : {"src", ".", "../src"}) {
    SCOPED_TRACE(Next);
    ASSERT_FALSE(Overlay->setCurrentWorkingDirectory(Next));
    auto Buffer = Overlay->getBufferForFile("header.h");
    ASSERT_TRUE(Buffer);
    EXPECT_EQ((*Buffer)->getBuffer(), "unsaved");
  }
}

#ifdef _WIN32
TEST(DraftStore, VFSWindowsRootedAndDriveRelativePaths) {
  DraftStore DS;
  DS.addDraft("C:/proj/header.h", "1", "unsaved");
  auto Base = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  ASSERT_TRUE(Base->addFile("C:/proj/header.h", 0,
                            llvm::MemoryBuffer::getMemBuffer("stale disk")));
  auto Overlay = llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(Base);
  auto DraftFS = DS.asVFS();
  Overlay->pushOverlay(DraftFS);
  ASSERT_FALSE(Overlay->setCurrentWorkingDirectory("C:/proj"));

  llvm::vfs::FileSystem *Systems[] = {DraftFS.get(), Overlay.get()};
  for (auto *FS : Systems) {
    for (const char *Path : {"header.h", "C:/proj/header.h", "/proj/header.h",
                             "\\proj\\header.h", "C:header.h", "c:header.h"}) {
      SCOPED_TRACE(Path);
      EXPECT_TRUE(FS->status(Path));
      auto Buffer = FS->getBufferForFile(Path);
      ASSERT_TRUE(Buffer);
      EXPECT_EQ((*Buffer)->getBuffer(), "unsaved");
    }
  }

  for (const char *Path : {"/proj", "\\proj", "C:", "c:"}) {
    SCOPED_TRACE(Path);
    ASSERT_FALSE(Overlay->setCurrentWorkingDirectory(Path));
    auto Buffer = Overlay->getBufferForFile("header.h");
    ASSERT_TRUE(Buffer);
    EXPECT_EQ((*Buffer)->getBuffer(), "unsaved");
  }
}
#endif

TEST(DraftStore, VFSDirectoryIterationComposesWithBase) {
  auto Base = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  ASSERT_TRUE(Base->addFile("/proj/on-disk.h", 0,
                            llvm::MemoryBuffer::getMemBuffer("disk")));
  ASSERT_TRUE(Base->addFile("/other/only-base.h", 0,
                            llvm::MemoryBuffer::getMemBuffer("disk")));

  DraftStore DS;
  DS.addDraft("/proj/dirty.h", "1", "dirty");
  auto Overlay = llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(Base);
  Overlay->pushOverlay(DS.asVFS());

  EXPECT_THAT(directoryEntries(*Overlay, "/proj"),
              UnorderedElementsAre("dirty.h", "on-disk.h"));
  EXPECT_THAT(directoryEntries(*Overlay, "/other"),
              UnorderedElementsAre("only-base.h"));
}

} // namespace
} // namespace clangd
} // namespace clang
