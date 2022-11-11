//===- unittests/Basic/FileEntryTest.cpp - Test FileEntry/FileEntryRef ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/FileEntry.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace clang {

class FileEntryTestHelper {
  StringMap<llvm::ErrorOr<FileEntryRef::MapValue>> Files;
  StringMap<llvm::ErrorOr<DirectoryEntry &>> Dirs;

  SmallVector<std::unique_ptr<FileEntry>, 5> FEs;
  SmallVector<std::unique_ptr<DirectoryEntry>, 5> DEs;
  DirectoryEntryRef DR;

public:
  FileEntryTestHelper() : DR(addDirectory("dir")) {}

  DirectoryEntryRef addDirectory(StringRef Name) {
    DEs.emplace_back(new DirectoryEntry());
    return DirectoryEntryRef(*Dirs.insert({Name, *DEs.back()}).first);
  }
  DirectoryEntryRef addDirectoryAlias(StringRef Name, DirectoryEntryRef Base) {
    return DirectoryEntryRef(
        *Dirs.insert({Name, const_cast<DirectoryEntry &>(Base.getDirEntry())})
             .first);
  }

  FileEntryRef addFile(StringRef Name) {
    FEs.emplace_back(new FileEntry());
    return FileEntryRef(
        *Files.insert({Name, FileEntryRef::MapValue(*FEs.back().get(), DR)})
             .first);
  }
  FileEntryRef addFileAlias(StringRef Name, FileEntryRef Base) {
    return FileEntryRef(
        *Files
             .insert(
                 {Name, FileEntryRef::MapValue(
                            const_cast<FileEntry &>(Base.getFileEntry()), DR)})
             .first);
  }
  FileEntryRef addFileRedirect(StringRef Name, FileEntryRef Base) {
    return FileEntryRef(
        *Files
             .insert({Name, FileEntryRef::MapValue(
                                const_cast<FileEntryRef::MapEntry &>(
                                    Base.getMapEntry()))})
             .first);
  }
};

namespace {
TEST(FileEntryTest, FileEntryRef) {
  FileEntryTestHelper Refs;
  FileEntryRef R1 = Refs.addFile("1");
  FileEntryRef R2 = Refs.addFile("2");
  FileEntryRef R1Also = Refs.addFileAlias("1-also", R1);
  FileEntryRef R1Redirect = Refs.addFileRedirect("1-redirect", R1);
  FileEntryRef R1Redirect2 = Refs.addFileRedirect("1-redirect2", R1Redirect);

  EXPECT_EQ("1", R1.getName());
  EXPECT_EQ("2", R2.getName());
  EXPECT_EQ("1-also", R1Also.getName());
  EXPECT_EQ("1", R1Redirect.getName());
  EXPECT_EQ("1", R1Redirect2.getName());

  EXPECT_EQ("1", R1.getNameAsRequested());
  EXPECT_EQ("1-redirect", R1Redirect.getNameAsRequested());
  EXPECT_EQ("1-redirect2", R1Redirect2.getNameAsRequested());

  EXPECT_NE(&R1.getFileEntry(), &R2.getFileEntry());
  EXPECT_EQ(&R1.getFileEntry(), &R1Also.getFileEntry());
  EXPECT_EQ(&R1.getFileEntry(), &R1Redirect.getFileEntry());
  EXPECT_EQ(&R1Redirect.getFileEntry(), &R1Redirect2.getFileEntry());

  const FileEntry *CE1 = R1;
  EXPECT_EQ(CE1, &R1.getFileEntry());
}

TEST(FileEntryTest, OptionalFileEntryRefDegradesToFileEntryPtr) {
  FileEntryTestHelper Refs;
  OptionalFileEntryRefDegradesToFileEntryPtr M0;
  OptionalFileEntryRefDegradesToFileEntryPtr M1 = Refs.addFile("1");
  OptionalFileEntryRefDegradesToFileEntryPtr M2 = Refs.addFile("2");
  OptionalFileEntryRefDegradesToFileEntryPtr M0Also = None;
  OptionalFileEntryRefDegradesToFileEntryPtr M1Also =
      Refs.addFileAlias("1-also", *M1);

  EXPECT_EQ(M0, M0Also);
  EXPECT_EQ(StringRef("1"), M1->getName());
  EXPECT_EQ(StringRef("2"), M2->getName());
  EXPECT_EQ(StringRef("1-also"), M1Also->getName());

  const FileEntry *CE1 = M1;
  EXPECT_EQ(CE1, &M1->getFileEntry());
}

TEST(FileEntryTest, equals) {
  FileEntryTestHelper Refs;
  FileEntryRef R1 = Refs.addFile("1");
  FileEntryRef R2 = Refs.addFile("2");
  FileEntryRef R1Also = Refs.addFileAlias("1-also", R1);
  FileEntryRef R1Redirect = Refs.addFileRedirect("1-redirect", R1);
  FileEntryRef R1Redirect2 = Refs.addFileRedirect("1-redirect2", R1Redirect);

  EXPECT_EQ(R1, &R1.getFileEntry());
  EXPECT_EQ(&R1.getFileEntry(), R1);
  EXPECT_EQ(R1, R1Also);
  EXPECT_NE(R1, &R2.getFileEntry());
  EXPECT_NE(&R2.getFileEntry(), R1);
  EXPECT_NE(R1, R2);
  EXPECT_EQ(R1, R1Redirect);
  EXPECT_EQ(R1, R1Redirect2);

  OptionalFileEntryRefDegradesToFileEntryPtr M1 = R1;

  EXPECT_EQ(M1, &R1.getFileEntry());
  EXPECT_EQ(&R1.getFileEntry(), M1);
  EXPECT_NE(M1, &R2.getFileEntry());
  EXPECT_NE(&R2.getFileEntry(), M1);
}

TEST(FileEntryTest, isSameRef) {
  FileEntryTestHelper Refs;
  FileEntryRef R1 = Refs.addFile("1");
  FileEntryRef R2 = Refs.addFile("2");
  FileEntryRef R1Also = Refs.addFileAlias("1-also", R1);
  FileEntryRef R1Redirect = Refs.addFileRedirect("1-redirect", R1);
  FileEntryRef R1Redirect2 = Refs.addFileRedirect("1-redirect2", R1Redirect);

  EXPECT_TRUE(R1.isSameRef(FileEntryRef(R1)));
  EXPECT_TRUE(R1.isSameRef(FileEntryRef(R1.getMapEntry())));
  EXPECT_FALSE(R1.isSameRef(R2));
  EXPECT_FALSE(R1.isSameRef(R1Also));
  EXPECT_FALSE(R1.isSameRef(R1Redirect));
  EXPECT_FALSE(R1.isSameRef(R1Redirect2));
  EXPECT_FALSE(R1Redirect.isSameRef(R1Redirect2));
}

TEST(FileEntryTest, DenseMapInfo) {
  FileEntryTestHelper Refs;
  FileEntryRef R1 = Refs.addFile("1");
  FileEntryRef R2 = Refs.addFile("2");
  FileEntryRef R1Also = Refs.addFileAlias("1-also", R1);

  // Insert R1Also first and confirm it "wins".
  {
    SmallDenseSet<FileEntryRef, 8> Set;
    Set.insert(R1Also);
    Set.insert(R1);
    Set.insert(R2);
    EXPECT_TRUE(Set.find(R1Also)->isSameRef(R1Also));
    EXPECT_TRUE(Set.find(R1)->isSameRef(R1Also));
    EXPECT_TRUE(Set.find(R2)->isSameRef(R2));
  }

  // Insert R1Also second and confirm R1 "wins".
  {
    SmallDenseSet<FileEntryRef, 8> Set;
    Set.insert(R1);
    Set.insert(R1Also);
    Set.insert(R2);
    EXPECT_TRUE(Set.find(R1Also)->isSameRef(R1));
    EXPECT_TRUE(Set.find(R1)->isSameRef(R1));
    EXPECT_TRUE(Set.find(R2)->isSameRef(R2));
  }
}

TEST(DirectoryEntryTest, isSameRef) {
  FileEntryTestHelper Refs;
  DirectoryEntryRef R1 = Refs.addDirectory("1");
  DirectoryEntryRef R2 = Refs.addDirectory("2");
  DirectoryEntryRef R1Also = Refs.addDirectoryAlias("1-also", R1);

  EXPECT_TRUE(R1.isSameRef(DirectoryEntryRef(R1)));
  EXPECT_TRUE(R1.isSameRef(DirectoryEntryRef(R1.getMapEntry())));
  EXPECT_FALSE(R1.isSameRef(R2));
  EXPECT_FALSE(R1.isSameRef(R1Also));
}

TEST(DirectoryEntryTest, DenseMapInfo) {
  FileEntryTestHelper Refs;
  DirectoryEntryRef R1 = Refs.addDirectory("1");
  DirectoryEntryRef R2 = Refs.addDirectory("2");
  DirectoryEntryRef R1Also = Refs.addDirectoryAlias("1-also", R1);

  // Insert R1Also first and confirm it "wins".
  {
    SmallDenseSet<DirectoryEntryRef, 8> Set;
    Set.insert(R1Also);
    Set.insert(R1);
    Set.insert(R2);
    EXPECT_TRUE(Set.find(R1Also)->isSameRef(R1Also));
    EXPECT_TRUE(Set.find(R1)->isSameRef(R1Also));
    EXPECT_TRUE(Set.find(R2)->isSameRef(R2));
  }

  // Insert R1Also second and confirm R1 "wins".
  {
    SmallDenseSet<DirectoryEntryRef, 8> Set;
    Set.insert(R1);
    Set.insert(R1Also);
    Set.insert(R2);
    EXPECT_TRUE(Set.find(R1Also)->isSameRef(R1));
    EXPECT_TRUE(Set.find(R1)->isSameRef(R1));
    EXPECT_TRUE(Set.find(R2)->isSameRef(R2));
  }
}

} // end namespace
} // namespace clang
