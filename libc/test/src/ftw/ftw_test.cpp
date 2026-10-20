//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for ftw.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "hdr/ftw_macros.h"
#include "hdr/types/struct_stat.h"
#include "src/__support/CPP/scope.h"
#include "src/__support/CPP/string_view.h"
#include "src/fcntl/open.h"
#include "src/ftw/ftw.h"
#include "src/sys/stat/chmod.h"
#include "src/sys/stat/mkdir.h"
#include "src/unistd/close.h"
#include "src/unistd/geteuid.h"
#include "src/unistd/rmdir.h"
#include "src/unistd/symlink.h"
#include "src/unistd/unlink.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

namespace {

int call_count = 0;
int dir_count = 0;
int file_count = 0;
int sl_count = 0;
int dnr_count = 0;
int ns_count = 0;

void reset_counts() {
  call_count = 0;
  dir_count = 0;
  file_count = 0;
  sl_count = 0;
  dnr_count = 0;
  ns_count = 0;
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

  // Verify traversal with trailing slash on root path:
  // Directories: root, dir1, and cycle_link (reported as FTW_D without
  // re-descending).
  // Regular files: file1.txt, file2.txt, and link_to_file1 (followed to FTW_F).
  // Dangling symlink: broken_link (reported as FTW_SL).
  ASSERT_THAT(LIBC_NAMESPACE::ftw(root_slash, record_callback, 10),
              Succeeds(0));
  EXPECT_EQ(call_count, 7);
  EXPECT_EQ(dir_count, 3);
  EXPECT_EQ(file_count, 3);
  EXPECT_EQ(sl_count, 1);

  // Verify fd_limit = 1 stops descending past root.
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

  EXPECT_THAT(LIBC_NAMESPACE::ftw("", record_callback, 10), Fails(ENOENT));
  EXPECT_THAT(LIBC_NAMESPACE::ftw("/nonexistent_ftw_path", record_callback, 10),
              Fails(ENOENT));
  EXPECT_THAT(LIBC_NAMESPACE::ftw(root, record_callback, 0), Fails(EINVAL));
  EXPECT_THAT(LIBC_NAMESPACE::ftw(root, record_callback, -1), Fails(EINVAL));
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

  // Calling ftw directly on a path whose parent lacks search permission must
  // fail with -1 and EACCES rather than reporting FTW_NS.
  EXPECT_THAT(LIBC_NAMESPACE::ftw(child, record_callback, 10), Fails(EACCES));
}

#ifdef LIBC_ADD_NULL_CHECKS
TEST_F(LlvmLibcFtwTest, NullptrCrash) {
  EXPECT_DEATH([] { LIBC_NAMESPACE::ftw(nullptr, record_callback, 10); },
               WITH_SIGNAL(-1));
  EXPECT_DEATH([] { LIBC_NAMESPACE::ftw(".", nullptr, 10); }, WITH_SIGNAL(-1));
}
#endif // LIBC_ADD_NULL_CHECKS
