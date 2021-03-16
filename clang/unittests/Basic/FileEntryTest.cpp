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
using namespace clang;

namespace {

using FileMap = StringMap<llvm::ErrorOr<FileEntryRef::MapValue>>;
using DirMap = StringMap<llvm::ErrorOr<DirectoryEntry &>>;

struct RefMaps {
  FileMap Files;
  DirMap Dirs;

  SmallVector<std::unique_ptr<FileEntry>, 5> FEs;
  SmallVector<std::unique_ptr<DirectoryEntry>, 5> DEs;
  DirectoryEntryRef DR;

  RefMaps() : DR(addDirectory("dir")) {}

  DirectoryEntryRef addDirectory(StringRef Name) {
    DEs.push_back(std::make_unique<DirectoryEntry>());
    return DirectoryEntryRef(*Dirs.insert({Name, *DEs.back()}).first);
  }
  DirectoryEntryRef addDirectoryAlias(StringRef Name, DirectoryEntryRef Base) {
    return DirectoryEntryRef(
        *Dirs.insert({Name, const_cast<DirectoryEntry &>(Base.getDirEntry())})
             .first);
  }

  FileEntryRef addFile(StringRef Name) {
    FEs.push_back(std::make_unique<FileEntry>());
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
};

TEST(FileEntryTest, Constructor) {
  FileEntry FE;
  EXPECT_EQ(0, FE.getSize());
  EXPECT_EQ(0, FE.getModificationTime());
  EXPECT_EQ(nullptr, FE.getDir());
  EXPECT_EQ(0U, FE.getUniqueID().getDevice());
  EXPECT_EQ(0U, FE.getUniqueID().getFile());
  EXPECT_EQ(false, FE.isNamedPipe());
  EXPECT_EQ(false, FE.isValid());
}

TEST(FileEntryTest, FileEntryRef) {
  RefMaps Refs;
  FileEntryRef R1 = Refs.addFile("1");
  FileEntryRef R2 = Refs.addFile("2");
  FileEntryRef R1Also = Refs.addFileAlias("1-also", R1);

  EXPECT_EQ("1", R1.getName());
  EXPECT_EQ("2", R2.getName());
  EXPECT_EQ("1-also", R1Also.getName());

  EXPECT_NE(&R1.getFileEntry(), &R2.getFileEntry());
  EXPECT_EQ(&R1.getFileEntry(), &R1Also.getFileEntry());

  const FileEntry *CE1 = R1;
  EXPECT_EQ(CE1, &R1.getFileEntry());
}

TEST(FileEntryTest, OptionalFileEntryRefDegradesToFileEntryPtr) {
  RefMaps Refs;
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
  RefMaps Refs;
  FileEntryRef R1 = Refs.addFile("1");
  FileEntryRef R2 = Refs.addFile("2");
  FileEntryRef R1Also = Refs.addFileAlias("1-also", R1);

  EXPECT_EQ(R1, &R1.getFileEntry());
  EXPECT_EQ(&R1.getFileEntry(), R1);
  EXPECT_EQ(R1, R1Also);
  EXPECT_NE(R1, &R2.getFileEntry());
  EXPECT_NE(&R2.getFileEntry(), R1);
  EXPECT_NE(R1, R2);

  OptionalFileEntryRefDegradesToFileEntryPtr M1 = R1;

  EXPECT_EQ(M1, &R1.getFileEntry());
  EXPECT_EQ(&R1.getFileEntry(), M1);
  EXPECT_NE(M1, &R2.getFileEntry());
  EXPECT_NE(&R2.getFileEntry(), M1);
}

TEST(FileEntryTest, isSameRef) {
  RefMaps Refs;
  FileEntryRef R1 = Refs.addFile("1");
  FileEntryRef R2 = Refs.addFile("2");
  FileEntryRef R1Also = Refs.addFileAlias("1-also", R1);

  EXPECT_TRUE(R1.isSameRef(FileEntryRef(R1)));
  EXPECT_TRUE(R1.isSameRef(FileEntryRef(R1.getMapEntry())));
  EXPECT_FALSE(R1.isSameRef(R2));
  EXPECT_FALSE(R1.isSameRef(R1Also));
}

TEST(FileEntryTest, DenseMapInfo) {
  RefMaps Refs;
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
  RefMaps Refs;
  DirectoryEntryRef R1 = Refs.addDirectory("1");
  DirectoryEntryRef R2 = Refs.addDirectory("2");
  DirectoryEntryRef R1Also = Refs.addDirectoryAlias("1-also", R1);

  EXPECT_TRUE(R1.isSameRef(DirectoryEntryRef(R1)));
  EXPECT_TRUE(R1.isSameRef(DirectoryEntryRef(R1.getMapEntry())));
  EXPECT_FALSE(R1.isSameRef(R2));
  EXPECT_FALSE(R1.isSameRef(R1Also));
}

TEST(DirectoryEntryTest, DenseMapInfo) {
  RefMaps Refs;
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
