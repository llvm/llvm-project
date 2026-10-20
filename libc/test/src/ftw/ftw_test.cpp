//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for ftw and nftw.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "hdr/ftw_macros.h"
#include "hdr/types/struct_FTW.h"
#include "hdr/types/struct_stat.h"
#include "src/__support/CPP/scope.h"
#include "src/__support/CPP/string_view.h"
#include "src/fcntl/open.h"
#include "src/ftw/ftw.h"
#include "src/ftw/nftw.h"
#include "src/sys/stat/chmod.h"
#include "src/sys/stat/mkdir.h"
#include "src/unistd/close.h"
#include "src/unistd/getcwd.h"
#include "src/unistd/geteuid.h"
#include "src/unistd/rmdir.h"
#include "src/unistd/symlink.h"
#include "src/unistd/unlink.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::cpp::string_view;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

namespace {

int call_count = 0;
int dir_count = 0;
int dp_count = 0;
int file_count = 0;
int sl_count = 0;
int sln_count = 0;
int dnr_count = 0;
int ns_count = 0;
int root_index = -1;
int subdir_index = -1;
int max_level = -1;
bool base_valid = true;
bool chdir_valid = true;

void reset_counts() {
  call_count = 0;
  dir_count = 0;
  dp_count = 0;
  file_count = 0;
  sl_count = 0;
  sln_count = 0;
  dnr_count = 0;
  ns_count = 0;
  root_index = -1;
  subdir_index = -1;
  max_level = -1;
  base_valid = true;
  chdir_valid = true;
}

int record_callback(const char *fpath, const struct stat *sb, int typeflag) {
  (void)fpath;
  (void)sb;
  ++call_count;
  if (typeflag == FTW_D)
    ++dir_count;
  else if (typeflag == FTW_F)
    ++file_count;
  else if (typeflag == FTW_SL)
    ++sl_count;
  else if (typeflag == FTW_DNR)
    ++dnr_count;
  else if (typeflag == FTW_NS)
    ++ns_count;
  return 0;
}

int stop_callback(const char *fpath, const struct stat *sb, int typeflag) {
  (void)fpath;
  (void)sb;
  (void)typeflag;
  ++call_count;
  return 42;
}

int nftw_record_callback(const char *fpath, const struct stat *sb, int typeflag,
                         struct FTW *ftwbuf) {
  (void)sb;
  string_view full(fpath);
  string_view base(fpath + ftwbuf->base);
  if (base.empty() || base.find_first_of('/') != string_view::npos)
    base_valid = false;
  if (ftwbuf->level > max_level)
    max_level = ftwbuf->level;
  if (ftwbuf->level == 0)
    root_index = call_count;
  if (base == "dir1")
    subdir_index = call_count;

  ++call_count;
  if (typeflag == FTW_D)
    ++dir_count;
  else if (typeflag == FTW_DP)
    ++dp_count;
  else if (typeflag == FTW_F)
    ++file_count;
  else if (typeflag == FTW_SL)
    ++sl_count;
  else if (typeflag == FTW_SLN)
    ++sln_count;
  return 0;
}

int nftw_chdir_callback(const char *fpath, const struct stat *sb, int typeflag,
                        struct FTW *ftwbuf) {
  (void)sb;
  ++call_count;
  if (ftwbuf->level > 0 && typeflag != FTW_SL && typeflag != FTW_SLN) {
    // When FTW_CHDIR is set, each child entry must be accessible via its
    // relative basename in the current working directory.
    const char *rel_name = fpath + ftwbuf->base;
    int fd = LIBC_NAMESPACE::open(rel_name, O_RDONLY);
    if (fd < 0)
      chdir_valid = false;
    else
      LIBC_NAMESPACE::close(fd);
  }
  return 0;
}

int nftw_skip_subtree_cb(const char *fpath, const struct stat *sb, int typeflag,
                         struct FTW *ftwbuf) {
  (void)sb;
  (void)typeflag;
  ++call_count;
  if (string_view(fpath + ftwbuf->base) == "dir1")
    return FTW_SKIP_SUBTREE;
  return FTW_CONTINUE;
}

int nftw_skip_siblings_cb(const char *fpath, const struct stat *sb,
                          int typeflag, struct FTW *ftwbuf) {
  (void)fpath;
  (void)sb;
  (void)typeflag;
  ++call_count;
  if (ftwbuf->level == 1)
    return FTW_SKIP_SIBLINGS;
  return FTW_CONTINUE;
}

int nftw_stop_action_cb(const char *fpath, const struct stat *sb, int typeflag,
                        struct FTW *ftwbuf) {
  (void)fpath;
  (void)sb;
  (void)typeflag;
  (void)ftwbuf;
  ++call_count;
  return FTW_STOP;
}

int nftw_neg_one_cb(const char *fpath, const struct stat *sb, int typeflag,
                    struct FTW *ftwbuf) {
  (void)fpath;
  (void)sb;
  (void)typeflag;
  (void)ftwbuf;
  return -1;
}

void create_empty_file(const char *path) {
  int fd = LIBC_NAMESPACE::open(path, O_CREAT | O_WRONLY | O_TRUNC, 0644);
  if (fd >= 0)
    LIBC_NAMESPACE::close(fd);
}

} // namespace

class LlvmLibcFtwTest : public LIBC_NAMESPACE::testing::ErrnoCheckingTest {
public:
  void SetUp() override {
    ErrnoCheckingTest::SetUp();
    reset_counts();
  }
};

TEST_F(LlvmLibcFtwTest, BasicAndSymlinkTraversal) {
  constexpr const char *ROOT_REL = "ftw_basic_root";
  constexpr const char *ROOT_SLASH_REL = "ftw_basic_root/";
  constexpr const char *SUBDIR_REL = "ftw_basic_root/dir1";
  constexpr const char *FILE1_REL = "ftw_basic_root/file1.txt";
  constexpr const char *FILE2_REL = "ftw_basic_root/dir1/file2.txt";
  constexpr const char *SYMLINK_REL = "ftw_basic_root/link_to_file1";
  constexpr const char *BROKEN_REL = "ftw_basic_root/broken_link";
  constexpr const char *CYCLE_REL = "ftw_basic_root/dir1/cycle_link";

  const char *root = libc_make_test_file_path(ROOT_REL);
  const char *root_slash = libc_make_test_file_path(ROOT_SLASH_REL);
  const char *subdir = libc_make_test_file_path(SUBDIR_REL);
  const char *file1 = libc_make_test_file_path(FILE1_REL);
  const char *file2 = libc_make_test_file_path(FILE2_REL);
  const char *symlink_path = libc_make_test_file_path(SYMLINK_REL);
  const char *broken_path = libc_make_test_file_path(BROKEN_REL);
  const char *cycle_path = libc_make_test_file_path(CYCLE_REL);

  ASSERT_THAT(LIBC_NAMESPACE::mkdir(root, 0755), Succeeds(0));
  auto cleanup = LIBC_NAMESPACE::cpp::scope_exit([&]() {
    LIBC_NAMESPACE::unlink(cycle_path);
    LIBC_NAMESPACE::unlink(broken_path);
    LIBC_NAMESPACE::unlink(symlink_path);
    LIBC_NAMESPACE::unlink(file2);
    LIBC_NAMESPACE::unlink(file1);
    LIBC_NAMESPACE::rmdir(subdir);
    LIBC_NAMESPACE::rmdir(root);
  });

  ASSERT_THAT(LIBC_NAMESPACE::mkdir(subdir, 0755), Succeeds(0));
  create_empty_file(file1);
  create_empty_file(file2);
  ASSERT_THAT(LIBC_NAMESPACE::symlink("file1.txt", symlink_path), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::symlink("nonexistent_target", broken_path),
              Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::symlink("..", cycle_path), Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::ftw(root_slash, record_callback, 10),
              Succeeds(0));
  EXPECT_EQ(call_count, 7);
  EXPECT_EQ(dir_count, 3);
  EXPECT_EQ(file_count, 3);
  EXPECT_EQ(sl_count, 1);

  reset_counts();
  ASSERT_THAT(LIBC_NAMESPACE::ftw(root, record_callback, 1), Succeeds(0));
  EXPECT_EQ(call_count, 1);
  EXPECT_EQ(dir_count, 1);
}

TEST_F(LlvmLibcFtwTest, CallbackReturnAndErrorCases) {
  constexpr const char *ROOT_REL = "ftw_err_root";
  const char *root = libc_make_test_file_path(ROOT_REL);
  ASSERT_THAT(LIBC_NAMESPACE::mkdir(root, 0755), Succeeds(0));
  auto cleanup =
      LIBC_NAMESPACE::cpp::scope_exit([&]() { LIBC_NAMESPACE::rmdir(root); });

  ASSERT_THAT(LIBC_NAMESPACE::ftw(root, stop_callback, 10), Succeeds(42));
  EXPECT_EQ(call_count, 1);

  // Returning -1 from nftw callback must NOT set errno.
  ASSERT_THAT(LIBC_NAMESPACE::nftw(root, nftw_neg_one_cb, 10, 0), Succeeds(-1));

  EXPECT_THAT(LIBC_NAMESPACE::ftw("", record_callback, 10), Fails(ENOENT));
  EXPECT_THAT(LIBC_NAMESPACE::nftw("", nftw_record_callback, 10, 0),
              Fails(ENOENT));
  EXPECT_THAT(LIBC_NAMESPACE::ftw("/nonexistent_ftw_path", record_callback, 10),
              Fails(ENOENT));
  EXPECT_THAT(LIBC_NAMESPACE::ftw(root, record_callback, 0), Fails(EINVAL));
  EXPECT_THAT(LIBC_NAMESPACE::nftw(root, nftw_record_callback, 0, 0),
              Fails(EINVAL));
}

TEST_F(LlvmLibcFtwTest, PermissionErrors) {
  if (LIBC_NAMESPACE::geteuid() == 0)
    return;

  constexpr const char *ROOT_REL = "ftw_perm_root";
  constexpr const char *UNREADABLE_REL = "ftw_perm_root/unreadable_dir";
  constexpr const char *NOSEARCH_REL = "ftw_perm_root/nosearch_dir";
  constexpr const char *CHILD_REL = "ftw_perm_root/nosearch_dir/child";

  const char *root = libc_make_test_file_path(ROOT_REL);
  const char *unreadable = libc_make_test_file_path(UNREADABLE_REL);
  const char *nosearch = libc_make_test_file_path(NOSEARCH_REL);
  const char *child = libc_make_test_file_path(CHILD_REL);

  ASSERT_THAT(LIBC_NAMESPACE::mkdir(root, 0755), Succeeds(0));
  auto cleanup = LIBC_NAMESPACE::cpp::scope_exit([&]() {
    LIBC_NAMESPACE::chmod(unreadable, 0755);
    LIBC_NAMESPACE::chmod(nosearch, 0755);
    LIBC_NAMESPACE::unlink(child);
    LIBC_NAMESPACE::rmdir(unreadable);
    LIBC_NAMESPACE::rmdir(nosearch);
    LIBC_NAMESPACE::rmdir(root);
  });

  ASSERT_THAT(LIBC_NAMESPACE::mkdir(unreadable, 0333), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::mkdir(nosearch, 0755), Succeeds(0));
  create_empty_file(child);
  ASSERT_THAT(LIBC_NAMESPACE::chmod(nosearch, 0666), Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::ftw(root, record_callback, 10), Succeeds(0));
  EXPECT_EQ(dnr_count, 1);
  EXPECT_EQ(ns_count, 1);

  EXPECT_THAT(LIBC_NAMESPACE::ftw(child, record_callback, 10), Fails(EACCES));
}

TEST_F(LlvmLibcFtwTest, NftwFlagsAndActionRetVal) {
  constexpr const char *ROOT_REL = "nftw_flags_root";
  constexpr const char *SUBDIR_REL = "nftw_flags_root/dir1";
  constexpr const char *FILE1_REL = "nftw_flags_root/file1.txt";
  constexpr const char *FILE2_REL = "nftw_flags_root/dir1/file2.txt";
  constexpr const char *SYMLINK_REL = "nftw_flags_root/link_to_file1";
  constexpr const char *BROKEN_REL = "nftw_flags_root/broken_link";

  const char *root = libc_make_test_file_path(ROOT_REL);
  const char *subdir = libc_make_test_file_path(SUBDIR_REL);
  const char *file1 = libc_make_test_file_path(FILE1_REL);
  const char *file2 = libc_make_test_file_path(FILE2_REL);
  const char *symlink_path = libc_make_test_file_path(SYMLINK_REL);
  const char *broken_path = libc_make_test_file_path(BROKEN_REL);

  ASSERT_THAT(LIBC_NAMESPACE::mkdir(root, 0755), Succeeds(0));
  auto cleanup = LIBC_NAMESPACE::cpp::scope_exit([&]() {
    LIBC_NAMESPACE::unlink(broken_path);
    LIBC_NAMESPACE::unlink(symlink_path);
    LIBC_NAMESPACE::unlink(file2);
    LIBC_NAMESPACE::unlink(file1);
    LIBC_NAMESPACE::rmdir(subdir);
    LIBC_NAMESPACE::rmdir(root);
  });

  ASSERT_THAT(LIBC_NAMESPACE::mkdir(subdir, 0755), Succeeds(0));
  create_empty_file(file1);
  create_empty_file(file2);
  ASSERT_THAT(LIBC_NAMESPACE::symlink("file1.txt", symlink_path), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::symlink("nonexistent_target", broken_path),
              Succeeds(0));

  // Default (symlink-following) mode: broken_link is reported as FTW_SLN,
  // link_to_file1 is followed as FTW_F.
  ASSERT_THAT(LIBC_NAMESPACE::nftw(root, nftw_record_callback, 10, FTW_MOUNT),
              Succeeds(0));
  EXPECT_TRUE(base_valid);
  EXPECT_EQ(max_level, 2);
  EXPECT_EQ(dir_count, 2);
  EXPECT_EQ(file_count, 3);
  EXPECT_EQ(sln_count, 1);
  EXPECT_EQ(sl_count, 0);

  // FTW_PHYS + FTW_DEPTH: both symlinks are FTW_SL, directories are FTW_DP,
  // and subdir is visited before root.
  reset_counts();
  ASSERT_THAT(LIBC_NAMESPACE::nftw(root, nftw_record_callback, 10,
                                   FTW_PHYS | FTW_DEPTH),
              Succeeds(0));
  EXPECT_TRUE(base_valid);
  EXPECT_EQ(dp_count, 2);
  EXPECT_EQ(sl_count, 2);
  EXPECT_GT(root_index, subdir_index);

  // FTW_CHDIR: verifies relative paths work during traversal and CWD is
  // restored afterwards.
  reset_counts();
  char cwd_before[512];
  char cwd_after[512];
  ASSERT_NE(LIBC_NAMESPACE::getcwd(cwd_before, sizeof(cwd_before)), nullptr);
  ASSERT_THAT(LIBC_NAMESPACE::nftw(root, nftw_chdir_callback, 10,
                                   FTW_CHDIR | FTW_PHYS | FTW_DEPTH),
              Succeeds(0));
  ASSERT_NE(LIBC_NAMESPACE::getcwd(cwd_after, sizeof(cwd_after)), nullptr);
  EXPECT_STREQ(cwd_before, cwd_after);
  EXPECT_TRUE(chdir_valid);

  // FTW_ACTIONRETVAL: FTW_SKIP_SUBTREE skips file2.txt inside dir1 (6 - 1 = 5).
  reset_counts();
  ASSERT_THAT(LIBC_NAMESPACE::nftw(root, nftw_skip_subtree_cb, 10,
                                   FTW_PHYS | FTW_ACTIONRETVAL),
              Succeeds(0));
  EXPECT_EQ(call_count, 5);

  // FTW_ACTIONRETVAL: FTW_SKIP_SIBLINGS at level 1 stops after the first
  // level-1 entry (plus its subtree if that entry was dir1).
  reset_counts();
  ASSERT_THAT(LIBC_NAMESPACE::nftw(root, nftw_skip_siblings_cb, 10,
                                   FTW_PHYS | FTW_ACTIONRETVAL),
              Succeeds(0));
  EXPECT_GE(call_count, 2);
  EXPECT_LE(call_count, 3);

  // FTW_ACTIONRETVAL: FTW_STOP halts immediately and returns FTW_STOP.
  reset_counts();
  ASSERT_THAT(
      LIBC_NAMESPACE::nftw(root, nftw_stop_action_cb, 10, FTW_ACTIONRETVAL),
      Succeeds(FTW_STOP));
  EXPECT_EQ(call_count, 1);
}

#ifdef LIBC_ADD_NULL_CHECKS
TEST_F(LlvmLibcFtwTest, NullptrCrash) {
  EXPECT_DEATH([] { LIBC_NAMESPACE::ftw(nullptr, record_callback, 10); },
               WITH_SIGNAL(-1));
  EXPECT_DEATH([] { LIBC_NAMESPACE::ftw(".", nullptr, 10); }, WITH_SIGNAL(-1));
  EXPECT_DEATH(
      [] { LIBC_NAMESPACE::nftw(nullptr, nftw_record_callback, 10, 0); },
      WITH_SIGNAL(-1));
  EXPECT_DEATH([] { LIBC_NAMESPACE::nftw(".", nullptr, 10, 0); },
               WITH_SIGNAL(-1));
}
#endif // LIBC_ADD_NULL_CHECKS
