//===-- Unittests for ftw and nftw ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/ftw_macros.h"
#include "src/__support/CPP/string_view.h"
#include "src/dirent/closedir.h"
#include "src/dirent/opendir.h"
#include "src/dirent/readdir.h"
#include "src/fcntl/open.h"
#ifdef LIBC_FULL_BUILD
#include "include/llvm-libc-types/struct_FTW.h"
#else
#include <ftw.h>
#endif
#include "src/ftw/ftw.h"
#include "src/ftw/nftw.h"
#include "src/sys/stat/chmod.h"
#include "src/sys/stat/mkdir.h"
#include "src/unistd/close.h"
#include "src/unistd/getcwd.h"
#include "src/unistd/rmdir.h"
#include "src/unistd/symlink.h"
#include "src/unistd/unlink.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE {

using LlvmLibcFtwTest = testing::ErrnoCheckingTest;
using LlvmLibcNftwTest = testing::ErrnoCheckingTest;
using cpp::string_view;

// Test data structure to track visited files
struct VisitedFiles {
  static constexpr int MAX_FILES = 32;
  char Paths[MAX_FILES][256];
  int Types[MAX_FILES];
  int Levels[MAX_FILES];
  int Count;

  void reset() { Count = 0; }

  void add(const char *Path, int Type, int Level) {
    if (Count < MAX_FILES) {
      // Copy path manually.
      int I = 0;
      while (Path[I] && I < 255) {
        Paths[Count][I] = Path[I];
        I++;
      }
      Paths[Count][I] = '\0';
      Types[Count] = Type;
      Levels[Count] = Level;
      Count++;
    }
  }

  bool contains(const char *Substring) const {
    string_view Sub(Substring);
    for (int I = 0; I < Count; I++) {
      string_view Path(Paths[I]);
      if (Path.find_first_of(Sub[0]) != string_view::npos) {
        // Simple Substring check
        for (size_t J = 0; J <= Path.size() - Sub.size(); J++) {
          bool Match = true;
          for (size_t K = 0; K < Sub.size() && Match; K++) {
            if (Paths[I][J + K] != Substring[K])
              Match = false;
          }
          if (Match)
            return true;
        }
      }
    }
    return false;
  }

  int getTypeFor(const char *Substring) const {
    string_view Sub(Substring);
    for (int I = 0; I < Count; I++) {
      string_view Path(Paths[I]);
      for (size_t J = 0; J + Sub.size() <= Path.size(); J++) {
        bool Match = true;
        for (size_t K = 0; K < Sub.size() && Match; K++) {
          if (Paths[I][J + K] != Substring[K])
            Match = false;
        }
        if (Match)
          return Types[I];
      }
    }
    return -1;
  }
};

static VisitedFiles gVisited;

// Callback for nftw that records visited files
static int recordVisit(const char *Fpath, const struct stat *Sb, int Typeflag,
                       struct FTW *Ftwbuf) {
  (void)Sb; // unused
  gVisited.add(Fpath, Typeflag, Ftwbuf->level);
  return 0; // continue traversal
}

// Callback for ftw that records visited files
static int recordVisitFtw(const char *Fpath, const struct stat *Sb,
                          int Typeflag) {
  (void)Sb; // unused
  gVisited.add(Fpath, Typeflag, 0);
  return 0; // continue traversal
}

// Simplest callback that does nothing
static int simpleCallback(const char *Fpath, const struct stat *Sb,
                          int Typeflag) {
  (void)Fpath;
  (void)Sb;
  (void)Typeflag;
  return 0;
}

// Use static test directory that exists
TEST_F(LlvmLibcFtwTest, BasicTraversalWithTestData) {
  // First make sure testdata directory exists
  ::DIR *Dir = LIBC_NAMESPACE::opendir(libc_make_test_file_path("testdata"));
  if (Dir == nullptr) {
    // Skip test if testdata doesn't exist
    return;
  }
  LIBC_NAMESPACE::closedir(Dir);

  int Result = LIBC_NAMESPACE::ftw(libc_make_test_file_path("testdata"),
                                   simpleCallback, 10);
  ASSERT_EQ(Result, 0);
}

TEST_F(LlvmLibcFtwTest, NonexistentPath) {
  gVisited.reset();
  int result = LIBC_NAMESPACE::ftw("/nonexistent/path", recordVisitFtw, 10);
  EXPECT_EQ(result, -1);
  ASSERT_ERRNO_EQ(ENOENT);
}

TEST_F(LlvmLibcNftwTest, BasicTraversalWithTestData) {
  gVisited.reset();
  int result = LIBC_NAMESPACE::nftw(libc_make_test_file_path("testdata"),
                                    recordVisit, 10, 0);
  ASSERT_EQ(result, 0);

  // Should have visited some files
  EXPECT_GE(gVisited.Count, 1);
}

TEST_F(LlvmLibcNftwTest, NonexistentPath) {
  gVisited.reset();
  int result = LIBC_NAMESPACE::nftw("/nonexistent/path/that/does/not/exist",
                                    recordVisit, 10, 0);
  EXPECT_EQ(result, -1);
  ASSERT_ERRNO_EQ(ENOENT);
}

TEST_F(LlvmLibcNftwTest, DepthFirstFlag) {
  gVisited.reset();
  int result = LIBC_NAMESPACE::nftw(libc_make_test_file_path("testdata"),
                                    recordVisit, 10, FTW_DEPTH);
  ASSERT_EQ(result, 0);

  // Verify post-order traversal: contents before directory
  int NestedIndex = -1;
  int SubdirIndex = -1;
  for (int i = 0; i < gVisited.Count; i++) {
    string_view Path(gVisited.Paths[i]);
    if (Path.ends_with("nested.txt"))
      NestedIndex = i;
    else if (Path.ends_with("subdir") && gVisited.Types[i] == FTW_DP)
      SubdirIndex = i;
  }
  ASSERT_NE(NestedIndex, -1);
  ASSERT_NE(SubdirIndex, -1);
  EXPECT_LT(NestedIndex, SubdirIndex);
}

TEST_F(LlvmLibcNftwTest, PhysicalFlag) {
  gVisited.reset();
  int result = LIBC_NAMESPACE::nftw(libc_make_test_file_path("testdata"),
                                    recordVisit, 10, FTW_PHYS);
  ASSERT_EQ(result, 0);

  // Should have visited files
  EXPECT_GE(gVisited.Count, 1);
}

TEST_F(LlvmLibcNftwTest, CallbackCanStopTraversal) {
  gVisited.reset();
  // Use a callback that returns non-zero
  auto stopImmediately = [](const char *, const struct stat *, int,
                            struct FTW *) -> int { return 42; };
  int result = LIBC_NAMESPACE::nftw(libc_make_test_file_path("testdata"),
                                    stopImmediately, 10, 0);
  // nftw should return the callback's return value
  EXPECT_EQ(result, 42);
}

TEST_F(LlvmLibcNftwTest, ChdirFlag) {
  char original_cwd[1024];
  ASSERT_TRUE(LIBC_NAMESPACE::getcwd(original_cwd, sizeof(original_cwd)) !=
              nullptr);

  auto checkCwd = [](const char *, const struct stat *, int,
                     struct FTW *) -> int {
    char cwd[1024];
    if (LIBC_NAMESPACE::getcwd(cwd, sizeof(cwd)) == nullptr)
      return -1;
    return 0;
  };

  int result = LIBC_NAMESPACE::nftw(libc_make_test_file_path("testdata"),
                                    checkCwd, 10, FTW_CHDIR);
  ASSERT_EQ(result, 0);

  char final_cwd[1024];
  ASSERT_TRUE(LIBC_NAMESPACE::getcwd(final_cwd, sizeof(final_cwd)) != nullptr);
  // Verify that the original CWD was restored
  ASSERT_STREQ(original_cwd, final_cwd);
}

TEST_F(LlvmLibcFtwTest, DanglingSymlinkMapping) {
  // Create a dangling symlink: link -> nonexistent
  auto linkName = libc_make_test_file_path("testdata/dangling_link");
  LIBC_NAMESPACE::unlink(linkName);
  ASSERT_EQ(LIBC_NAMESPACE::symlink("nonexistent_target", linkName), 0);

  auto checkFtw = [](const char *Fpath, const struct stat *,
                     int Typeflag) -> int {
    if (string_view(Fpath).ends_with("dangling_link")) {
      // For legacy ftw, FTW_SLN must be mapped to FTW_SL
      if (Typeflag == FTW_SL)
        return 0;
      return -1;
    }
    return 0;
  };

  libc_errno = 0;
  int result = LIBC_NAMESPACE::ftw(linkName, checkFtw, 10);
  EXPECT_EQ(result, 0);

  auto checkNftw = [](const char *Fpath, const struct stat *, int Typeflag,
                      struct FTW *) -> int {
    if (string_view(Fpath).ends_with("dangling_link")) {
      // For nftw, FTW_SLN should be reported as is
      if (Typeflag == FTW_SLN)
        return 0;
      return -1;
    }
    return 0;
  };

  libc_errno = 0;
  result = LIBC_NAMESPACE::nftw(linkName, checkNftw, 10, 0);
  EXPECT_EQ(result, 0);

  LIBC_NAMESPACE::unlink(linkName);
  libc_errno = 0;
}

TEST_F(LlvmLibcNftwTest, UnreadableDirectory) {
  // Create an unreadable directory
  const char *dirName = "unreadable_dir";
  LIBC_NAMESPACE::rmdir(dirName);
  ASSERT_EQ(LIBC_NAMESPACE::mkdir(dirName, 0333), 0); // No read permission

  gVisited.reset();
  int result = LIBC_NAMESPACE::nftw(dirName, recordVisit, 10, 0);
  EXPECT_EQ(result, 0);

  // Should have visited the directory itself as FTW_DNR
  bool found = false;
  for (int i = 0; i < gVisited.Count; i++) {
    if (string_view(gVisited.Paths[i]) == dirName) {
      EXPECT_EQ(gVisited.Types[i], FTW_DNR);
      found = true;
    }
  }
  EXPECT_TRUE(found);

  LIBC_NAMESPACE::rmdir(dirName);
  libc_errno = 0;
}

TEST_F(LlvmLibcNftwTest, NoSearchPermission) {
  // Create a parent directory and a child, then remove search permission from
  // parent
  const char *parentName = "no_search_parent";
  const char *childName = "no_search_parent/child";

  LIBC_NAMESPACE::unlink(childName);
  LIBC_NAMESPACE::rmdir(parentName);

  ASSERT_EQ(LIBC_NAMESPACE::mkdir(parentName, 0777), 0);
  int fd = LIBC_NAMESPACE::open(childName, O_CREAT | O_WRONLY, 0666);
  ASSERT_GE(fd, 0);
  LIBC_NAMESPACE::close(fd);

  // Remove search (execute) permission from parent
  ASSERT_EQ(LIBC_NAMESPACE::chmod(parentName, 0666), 0);

  gVisited.reset();
  // We specify FTW_PHYS to avoid stat() trying to resolve and potentially
  // failing with EACCES before nftw handles it
  int result = LIBC_NAMESPACE::nftw(childName, recordVisit, 10, FTW_PHYS);
  EXPECT_EQ(result, 0);

  // Should have visited the child as FTW_NS
  bool found = false;
  for (int i = 0; i < gVisited.Count; i++) {
    if (string_view(gVisited.Paths[i]) == childName) {
      EXPECT_EQ(gVisited.Types[i], FTW_NS);
      found = true;
    }
  }
  EXPECT_TRUE(found);

  // Restore permission to allow cleanup
  LIBC_NAMESPACE::chmod(parentName, 0777);
  LIBC_NAMESPACE::unlink(childName);
  LIBC_NAMESPACE::rmdir(parentName);
  libc_errno = 0;
}

TEST_F(LlvmLibcNftwTest, ExcessiveDepthRespectsFdLimit) {
  // Creating a path with depth > 2
  auto path = libc_make_test_file_path("testdata");
  // If we specify fdLimit = 1, it should fail with EMFILE when trying to
  // iterate subdir, or even when visiting files in testdata because of how
  // recursion works. level 0 (testdata): fdLimit=1. Continue. level 1
  // (file1.txt): doMergedFtw(..., fdLimit=0). FAILS!
  int result = LIBC_NAMESPACE::nftw(path, recordVisit, 1, 0);
  EXPECT_EQ(result, -1);
  ASSERT_ERRNO_EQ(EMFILE);
}
TEST_F(LlvmLibcNftwTest, ActionRetValSkipSubtree) {
  gVisited.reset();
  auto callback = [](const char *Fpath, const struct stat *, int Typeflag,
                     struct FTW *Ftwbuf) -> int {
    gVisited.add(Fpath, Typeflag, Ftwbuf->level);
    if (string_view(Fpath).ends_with("subdir"))
      return FTW_SKIP_SUBTREE;
    return FTW_CONTINUE;
  };

  int result = LIBC_NAMESPACE::nftw(libc_make_test_file_path("testdata"),
                                    callback, 10, FTW_ACTIONRETVAL);
  ASSERT_EQ(result, 0);

  bool FoundSubdir = false;
  bool FoundNested = false;
  for (int i = 0; i < gVisited.Count; i++) {
    string_view Path(gVisited.Paths[i]);
    if (Path.ends_with("subdir"))
      FoundSubdir = true;
    if (Path.ends_with("nested.txt"))
      FoundNested = true;
  }
  EXPECT_TRUE(FoundSubdir);
  EXPECT_FALSE(FoundNested);
}

TEST_F(LlvmLibcNftwTest, ActionRetValSkipSiblings) {
  gVisited.reset();
  auto callback = [](const char *Fpath, const struct stat *, int Typeflag,
                     struct FTW *Ftwbuf) -> int {
    gVisited.add(Fpath, Typeflag, Ftwbuf->level);
    if (Ftwbuf->level == 1)
      return FTW_SKIP_SIBLINGS;
    return FTW_CONTINUE;
  };

  int result = LIBC_NAMESPACE::nftw(libc_make_test_file_path("testdata"),
                                    callback, 10, FTW_ACTIONRETVAL);
  ASSERT_EQ(result, 0);

  int Level1Count = 0;
  for (int i = 0; i < gVisited.Count; i++) {
    if (gVisited.Levels[i] == 1)
      Level1Count++;
  }
  EXPECT_EQ(Level1Count, 1);
}

} // namespace LIBC_NAMESPACE
