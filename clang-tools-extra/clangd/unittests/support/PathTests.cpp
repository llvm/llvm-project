//===-- PathTests.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestFS.h"
#include "support/Path.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

TEST(PathTests, HasWindowsDrive) {
  for (llvm::StringRef P : {"C:", "c:foo", "C:/foo", "z:\\foo"})
    EXPECT_TRUE(hasWindowsDrive(P)) << P;
  for (llvm::StringRef P : {"", "C", ":", "1:/foo", "/C:/foo",
                            "\\\\server\\share", "/foo", "file:///C:/foo"})
    EXPECT_FALSE(hasWindowsDrive(P)) << P;
}

TEST(PathTests, IsAncestor) {
  EXPECT_TRUE(PathRef(testPath("foo")).isAncestorOf(testPath("foo")));
  EXPECT_TRUE(PathRef(testPath("foo/")).isAncestorOf(testPath("foo")));

  EXPECT_FALSE(PathRef(testPath("foo")).isAncestorOf(testPath("fooz")));
  EXPECT_FALSE(PathRef(testPath("foo/")).isAncestorOf(testPath("fooz")));

  EXPECT_TRUE(PathRef(testPath("foo")).isAncestorOf(testPath("foo/bar")));
  EXPECT_TRUE(PathRef(testPath("foo/")).isAncestorOf(testPath("foo/bar")));

#ifdef CLANGD_PATH_CASE_INSENSITIVE
  EXPECT_TRUE(PathRef(testPath("fOo")).isAncestorOf(testPath("foo/bar")));
  EXPECT_TRUE(PathRef(testPath("foo")).isAncestorOf(testPath("fOo/bar")));
#else
  EXPECT_FALSE(PathRef(testPath("fOo")).isAncestorOf(testPath("foo/bar")));
  EXPECT_FALSE(PathRef(testPath("foo")).isAncestorOf(testPath("fOo/bar")));
#endif
}

TEST(PathTests, PosixSeparatorsWithNativeRoots) {
  const auto Posix = llvm::sys::path::Style::posix;
  auto Parent = testPath("proj", Posix);
  EXPECT_TRUE(
      PathRef(Parent).isAncestorOf(testPath("proj/a.cpp", Posix), Posix));
  EXPECT_FALSE(
      PathRef(Parent).isAncestorOf(testPath("project/a.cpp", Posix), Posix));
}

TEST(PathTests, DriveLetterIdentity) {
  Path Upper("C:/Users/src/foo.cpp");
  Path Lower("c:/Users/src/foo.cpp");
  EXPECT_EQ(PathRef(Upper), PathRef(Lower));
  EXPECT_TRUE(pathEqualLegacyCaseFold(Upper, Lower));
  EXPECT_EQ(pathHash(Upper.raw()), pathHash(Lower.raw()));
  EXPECT_NE(PathRef("C:/Users/src/foo.cpp"), PathRef("D:/Users/src/foo.cpp"));

#ifndef CLANGD_PATH_CASE_INSENSITIVE
  // Only the drive letter is folded on case-sensitive hosts.
  EXPECT_NE(PathRef("C:/Users/src/Foo.cpp"), PathRef("c:/Users/src/foo.cpp"));
#endif
}

TEST(PathTests, WindowsSlashIdentity) {
  EXPECT_EQ(PathRef("C:/proj/a.cpp"), PathRef("C:\\proj\\a.cpp"));
  EXPECT_EQ(PathRef("C:/proj/a.cpp"), PathRef("c:\\proj\\a.cpp"));
  EXPECT_EQ(pathHash("C:/proj/a.cpp"), pathHash("c:\\proj\\a.cpp"));
  EXPECT_EQ(PathRef("C:/proj/a.cpp").caseFolded().raw(), "c:/proj/a.cpp");
  EXPECT_EQ(PathRef("C:\\proj\\a.cpp").caseFolded().raw(), "c:/proj/a.cpp");
}

TEST(PathTests, WindowsPathClassification) {
#ifndef CLANGD_PATH_CASE_INSENSITIVE
  // A drive-relative POSIX filename is not an absolute Windows drive path.
  EXPECT_NE(PathRef("C:notes"), PathRef("c:notes"));
#endif

  EXPECT_EQ(PathRef("//server/share/a.cpp"),
            PathRef("\\\\server\\share\\a.cpp"));
  EXPECT_EQ(pathHash("//server/share/a.cpp"),
            pathHash("\\\\server\\share\\a.cpp"));
}

TEST(PathTests, RemoveDotsUsesPathStyle) {
  EXPECT_EQ(PathRef("C:\\proj\\src\\..\\a.cpp").removeDots(),
            Path("C:\\proj\\a.cpp"));
  EXPECT_EQ(PathRef("\\\\server\\share\\src\\..\\a.cpp").removeDots(),
            Path("\\\\server\\share\\a.cpp"));
}

TEST(PathTests, DriveLetterAncestor) {
  const auto Win = llvm::sys::path::Style::windows;
  EXPECT_TRUE(PathRef("C:/").isAncestorOf("c:/proj/a.cpp", Win));
  EXPECT_TRUE(PathRef("C:\\").isAncestorOf("c:/proj/a.cpp", Win));
  EXPECT_TRUE(PathRef("c:/").isAncestorOf("C:\\", Win));
  EXPECT_FALSE(PathRef("C:/").isAncestorOf("d:/proj/a.cpp", Win));
  EXPECT_TRUE(PathRef("C:/proj").isAncestorOf("c:/proj/src/a.cpp", Win));
  EXPECT_TRUE(PathRef("c:/proj/").isAncestorOf("C:/proj", Win));
  EXPECT_TRUE(PathRef("C:/proj").isAncestorOf("c:\\proj\\src\\a.cpp", Win));
  EXPECT_FALSE(PathRef("C:/proj").isAncestorOf("c:/other/a.cpp", Win));
}

TEST(PathTests, PathMapDriveLetter) {
  PathMap<int> M;
  M[PathRef("C:/proj/a.cpp")] = 1;
  auto It = M.find(PathRef("c:/proj/a.cpp"));
  ASSERT_NE(It, M.end());
  EXPECT_EQ(It->second, 1);
  // First-inserted spelling is preserved.
  EXPECT_EQ(It->first.raw(), "C:/proj/a.cpp");
  EXPECT_TRUE(M.contains(PathRef("c:/proj/a.cpp")));
  EXPECT_EQ(M[PathRef("c:/proj/a.cpp")], 1);
  EXPECT_EQ(M.size(), 1u);

  auto [InsertedIt, Inserted] = M.try_emplace(PathRef("c:/proj/a.cpp"), 2);
  EXPECT_FALSE(Inserted);
  EXPECT_EQ(InsertedIt->second, 1);

  EXPECT_TRUE(M.erase(PathRef("c:/proj/a.cpp")));
  EXPECT_TRUE(M.empty());
}

TEST(PathTests, PathMapEraseIterator) {
  PathMap<int> M;
  M["C:/proj/a.cpp"] = 1;
  M["C:/proj/b.cpp"] = 2;
  auto It = M.find("c:/proj/a.cpp");
  ASSERT_NE(It, M.end());
  M.erase(It);
  EXPECT_FALSE(M.contains("C:/proj/a.cpp"));
  EXPECT_EQ(M.lookup("C:/proj/b.cpp"), 2);
  EXPECT_EQ(M.size(), 1u);
  M.erase(M.begin());
  EXPECT_TRUE(M.empty());
}

TEST(PathTests, PathMapPreservesFilenameCase) {
  PathMap<int> M;
  M[PathRef("C:/Proj/A.cpp")] = 7;
  M[PathRef("C:/Proj/a.cpp")] = 8;
  EXPECT_EQ(M.size(), 2u);
  EXPECT_EQ(M.lookup(PathRef("c:/Proj/A.cpp")), 7);
  EXPECT_EQ(M.lookup(PathRef("c:/Proj/a.cpp")), 8);
  EXPECT_FALSE(M.contains(PathRef("C:/proj/A.cpp")));
}

TEST(PathTests, IdentityNormalizationAndOrdering) {
  llvm::StringRef Paths[] = {"C:/Proj/A.cpp",
                             "c:\\Proj\\A.cpp",
                             "c:/Proj/a.cpp",
                             "C:notes",
                             "c:notes",
                             "//server/share/a",
                             "\\\\server\\share\\a",
                             "relative/file",
                             "relative\\file",
                             "/",
                             ""};
  for (auto L : Paths) {
    for (auto R : Paths) {
      SCOPED_TRACE(L.str() + " vs " + R.str());
      auto LNorm = PathRef(L).identityNormalized();
      auto RNorm = PathRef(R).identityNormalized();
      EXPECT_EQ(pathEquals(L, R), LNorm.raw() == RNorm.raw());
      EXPECT_EQ(pathCompare(L, R) == 0, pathEquals(L, R));
      EXPECT_EQ(pathCompare(L, R) < 0, LNorm.raw() < RNorm.raw());
      if (pathEquals(L, R))
        EXPECT_EQ(pathHash(L), pathHash(R));
    }
  }
  EXPECT_EQ(PathRef("C:\\Proj\\A.cpp").identityNormalized().raw(),
            "c:/Proj/A.cpp");
}

} // namespace
} // namespace clangd
} // namespace clang
