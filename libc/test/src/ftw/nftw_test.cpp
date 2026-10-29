//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for nftw.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "hdr/ftw_macros.h"
#include "hdr/sys_stat_macros.h"
#include "hdr/types/mode_t.h"
#include "hdr/types/struct_FTW.h"
#include "hdr/types/struct_stat.h"
#include "src/__support/CPP/scope.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/libc_errno.h"
#include "src/fcntl/open.h"
#include "src/ftw/nftw.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "src/sys/stat/chmod.h"
#include "src/sys/stat/mkdir.h"
#include "src/unistd/chdir.h"
#include "src/unistd/close.h"
#include "src/unistd/getcwd.h"
#include "src/unistd/geteuid.h"
#include "src/unistd/rmdir.h"
#include "src/unistd/symlink.h"
#include "src/unistd/unlink.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"
#include "test/UnitTest/TestLogger.h"

using LIBC_NAMESPACE::cpp::string_view;
using LIBC_NAMESPACE::testing::tlog;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

namespace {

struct Visit {
  char path[256];
  size_t path_len;
  int typeflag;
  int level;
  int base;
  mode_t fmt;
};

constexpr size_t MAX_VISITS = 64;
Visit visits[MAX_VISITS];
size_t visit_count = 0;

int call_count = 0;
int dir_count = 0;
int dp_count = 0;
int file_count = 0;
int sl_count = 0;
int sln_count = 0;
int dnr_count = 0;
int ns_count = 0;
int max_level = -1;

void reset_counts() {
  visit_count = 0;
  call_count = 0;
  dir_count = 0;
  dp_count = 0;
  file_count = 0;
  sl_count = 0;
  sln_count = 0;
  dnr_count = 0;
  ns_count = 0;
  max_level = -1;
}

int find_visit(string_view path) {
  for (size_t i = 0; i < visit_count; ++i)
    if (string_view(visits[i].path, visits[i].path_len) == path)
      return static_cast<int>(i);
  return -1;
}

int record_callback(const char *fpath, const struct stat *sb, int typeflag,
                    struct FTW *ftwbuf) {
  string_view full(fpath);
  string_view base(fpath + ftwbuf->base);
  EXPECT_GT(base.size(), size_t(0));
  EXPECT_EQ(base.find_first_of('/'), string_view::npos);
  if (ftwbuf->base > 0)
    EXPECT_EQ(fpath[ftwbuf->base - 1], '/');
  if (ftwbuf->base > 1)
    EXPECT_NE(fpath[ftwbuf->base - 2], '/');

  if (visit_count < MAX_VISITS && full.size() < sizeof(visits[0].path)) {
    Visit &v = visits[visit_count++];
    LIBC_NAMESPACE::inline_memcpy(v.path, full.data(), full.size());
    v.path[full.size()] = '\0';
    v.path_len = full.size();
    v.typeflag = typeflag;
    v.level = ftwbuf->level;
    v.base = ftwbuf->base;
    v.fmt = sb->st_mode & S_IFMT;
  }

  if (ftwbuf->level > max_level)
    max_level = ftwbuf->level;

  ++call_count;
  if (typeflag == FTW_D) {
    EXPECT_EQ(sb->st_mode & S_IFMT, static_cast<mode_t>(S_IFDIR));
    ++dir_count;
  } else if (typeflag == FTW_DP) {
    EXPECT_EQ(sb->st_mode & S_IFMT, static_cast<mode_t>(S_IFDIR));
    ++dp_count;
  } else if (typeflag == FTW_F) {
    EXPECT_EQ(sb->st_mode & S_IFMT, static_cast<mode_t>(S_IFREG));
    ++file_count;
  } else if (typeflag == FTW_SL) {
    EXPECT_EQ(sb->st_mode & S_IFMT, static_cast<mode_t>(S_IFLNK));
    ++sl_count;
  } else if (typeflag == FTW_SLN) {
    EXPECT_EQ(sb->st_mode & S_IFMT, static_cast<mode_t>(S_IFLNK));
    ++sln_count;
  } else if (typeflag == FTW_DNR) {
    EXPECT_EQ(sb->st_mode & S_IFMT, static_cast<mode_t>(S_IFDIR));
    ++dnr_count;
  } else if (typeflag == FTW_NS) {
    ++ns_count;
  }
  return 0;
}

int chdir_callback(const char *fpath, const struct stat *sb, int typeflag,
                   struct FTW *ftwbuf) {
  (void)sb;
  ++call_count;
  if (typeflag != FTW_SL && typeflag != FTW_SLN) {
    // When FTW_CHDIR is set, every entry (including level 0 when the root path
    // contains a directory prefix) must resolve via its relative basename in
    // the current working directory.
    const char *rel_name = fpath + ftwbuf->base;
    int fd = LIBC_NAMESPACE::open(rel_name, O_RDONLY);
    EXPECT_GE(fd, 0);
    if (fd >= 0)
      EXPECT_EQ(LIBC_NAMESPACE::close(fd), 0);
  }
  return 0;
}

int unlink_sibling_cb(const char *fpath, const struct stat *sb, int typeflag,
                      struct FTW *ftwbuf) {
  (void)sb;
  ++call_count;
  if (typeflag == FTW_NS)
    ++ns_count;
  string_view base(fpath + ftwbuf->base);
  if (typeflag == FTW_F && base == "vanish_a.txt")
    LIBC_NAMESPACE::unlink("nftw_race_root/vanish_b.txt");
  else if (typeflag == FTW_F && base == "vanish_b.txt")
    LIBC_NAMESPACE::unlink("nftw_race_root/vanish_a.txt");
  return 0;
}

int stop_42_cb(const char *fpath, const struct stat *sb, int typeflag,
               struct FTW *ftwbuf) {
  (void)fpath;
  (void)sb;
  (void)typeflag;
  (void)ftwbuf;
  ++call_count;
  return 42;
}

int stop_neg_one_cb(const char *fpath, const struct stat *sb, int typeflag,
                    struct FTW *ftwbuf) {
  (void)fpath;
  (void)sb;
  (void)typeflag;
  (void)ftwbuf;
  ++call_count;
  LIBC_NAMESPACE::libc_errno = EXDEV;
  return -1;
}

int stop_at_level1_cb(const char *fpath, const struct stat *sb, int typeflag,
                      struct FTW *ftwbuf) {
  (void)fpath;
  (void)sb;
  (void)typeflag;
  ++call_count;
  if (ftwbuf->level == 1)
    return 99;
  return 0;
}

int skip_subtree_cb(const char *fpath, const struct stat *sb, int typeflag,
                    struct FTW *ftwbuf) {
  (void)sb;
  (void)typeflag;
  ++call_count;
  if (string_view(fpath + ftwbuf->base) == "dir1")
    return FTW_SKIP_SUBTREE;
  return FTW_CONTINUE;
}

int skip_siblings_cb(const char *fpath, const struct stat *sb, int typeflag,
                     struct FTW *ftwbuf) {
  (void)fpath;
  (void)sb;
  (void)typeflag;
  ++call_count;
  if (ftwbuf->level == 1)
    return FTW_SKIP_SIBLINGS;
  return FTW_CONTINUE;
}

int stop_action_cb(const char *fpath, const struct stat *sb, int typeflag,
                   struct FTW *ftwbuf) {
  (void)fpath;
  (void)sb;
  (void)typeflag;
  (void)ftwbuf;
  ++call_count;
  return FTW_STOP;
}

int create_empty_file(const char *path) {
  int fd = LIBC_NAMESPACE::open(path, O_CREAT | O_WRONLY | O_TRUNC, 0644);
  if (fd < 0)
    return -1;
  return LIBC_NAMESPACE::close(fd);
}

} // namespace

class LlvmLibcNftwTest : public LIBC_NAMESPACE::testing::ErrnoCheckingTest {
public:
  void SetUp() override {
    ErrnoCheckingTest::SetUp();
    reset_counts();
  }
};

TEST_F(LlvmLibcNftwTest, BasicAndSymlinkTraversal) {
  constexpr const char *ROOT_REL = "nftw_basic_root";
  constexpr const char *ROOT_SLASH_REL = "nftw_basic_root/";
  constexpr const char *SUBDIR_REL = "nftw_basic_root/dir1";
  constexpr const char *FILE1_REL = "nftw_basic_root/file1.txt";
  constexpr const char *FILE2_REL = "nftw_basic_root/dir1/file2.txt";
  constexpr const char *SYMLINK_REL = "nftw_basic_root/link_to_file1";
  constexpr const char *BROKEN_REL = "nftw_basic_root/broken_link";
  constexpr const char *CYCLE_REL = "nftw_basic_root/dir1/cycle_link";
  constexpr const char *DUPDIR_REL = "nftw_basic_root/dup_dir1";

  auto root = libc_make_test_file_path(ROOT_REL);
  auto root_slash = libc_make_test_file_path(ROOT_SLASH_REL);
  auto subdir = libc_make_test_file_path(SUBDIR_REL);
  auto file1 = libc_make_test_file_path(FILE1_REL);
  auto file2 = libc_make_test_file_path(FILE2_REL);
  auto symlink_path = libc_make_test_file_path(SYMLINK_REL);
  auto broken_path = libc_make_test_file_path(BROKEN_REL);
  auto cycle_path = libc_make_test_file_path(CYCLE_REL);
  auto dupdir_path = libc_make_test_file_path(DUPDIR_REL);

  ASSERT_THAT(LIBC_NAMESPACE::mkdir(root, 0755), Succeeds(0));
  auto cleanup = LIBC_NAMESPACE::cpp::scope_exit([&]() {
    LIBC_NAMESPACE::unlink(dupdir_path);
    LIBC_NAMESPACE::unlink(cycle_path);
    LIBC_NAMESPACE::unlink(broken_path);
    LIBC_NAMESPACE::unlink(symlink_path);
    LIBC_NAMESPACE::unlink(file2);
    LIBC_NAMESPACE::unlink(file1);
    LIBC_NAMESPACE::rmdir(subdir);
    LIBC_NAMESPACE::rmdir(root);
  });

  ASSERT_THAT(LIBC_NAMESPACE::mkdir(subdir, 0755), Succeeds(0));
  ASSERT_THAT(create_empty_file(file1), Succeeds(0));
  ASSERT_THAT(create_empty_file(file2), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::symlink("file1.txt", symlink_path), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::symlink("nonexistent_target", broken_path),
              Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::symlink("..", cycle_path), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::symlink("dir1", dupdir_path), Succeeds(0));

  // POSIX.1-2024 FTW_PHYS: "If clear, nftw() shall follow links instead of
  // reporting them, and shall not report the same file twice." Both cycle_link
  // (pointing to ancestor root) and whichever of dir1 / dup_dir1 is visited
  // second are omitted, leaving 6 unique visits.
  ASSERT_THAT(LIBC_NAMESPACE::nftw(root_slash, record_callback, 10, 0),
              Succeeds(0));
  EXPECT_EQ(call_count, 6);
  EXPECT_EQ(dir_count, 2);
  EXPECT_EQ(file_count, 3);
  EXPECT_EQ(sln_count, 1);
  EXPECT_EQ(sl_count, 0);

  int root_idx = find_visit(ROOT_REL);
  int f1_idx = find_visit(FILE1_REL);
  int broken_idx = find_visit(BROKEN_REL);
  ASSERT_EQ(root_idx, 0);
  EXPECT_EQ(visits[root_idx].typeflag, FTW_D);
  EXPECT_EQ(visits[root_idx].level, 0);
  ASSERT_GE(f1_idx, 0);
  EXPECT_EQ(visits[f1_idx].typeflag, FTW_F);
  EXPECT_EQ(visits[f1_idx].level, 1);
  ASSERT_GE(broken_idx, 0);
  EXPECT_EQ(visits[broken_idx].typeflag, FTW_SLN);
  EXPECT_EQ(visits[broken_idx].level, 1);
  EXPECT_EQ(find_visit(CYCLE_REL), -1);

  // Logical post-order walk (FTW_DEPTH without FTW_PHYS): omits duplicate
  // directories and reports directories after their descendants as FTW_DP.
  reset_counts();
  ASSERT_THAT(LIBC_NAMESPACE::nftw(root, record_callback, 10, FTW_DEPTH),
              Succeeds(0));
  EXPECT_EQ(call_count, 6);
  EXPECT_EQ(dp_count, 2);
  EXPECT_EQ(file_count, 3);
  EXPECT_EQ(sln_count, 1);
  EXPECT_EQ(find_visit(ROOT_REL), 5);
}

TEST_F(LlvmLibcNftwTest, ConstrainedFdLimitAndChdir) {
  constexpr const char *ROOT_REL = "nftw_fdlimit_root";
  constexpr const char *DIR1_REL = "nftw_fdlimit_root/dir1";
  constexpr const char *DIR2_REL = "nftw_fdlimit_root/dir1/dir2";
  constexpr const char *FILE3_REL = "nftw_fdlimit_root/dir1/dir2/file3.txt";
  constexpr const char *DIRB_REL = "nftw_fdlimit_root/dir1/dirb";
  constexpr const char *FILEB_REL = "nftw_fdlimit_root/dir1/dirb/fileb.txt";

  auto root = libc_make_test_file_path(ROOT_REL);
  auto dir1 = libc_make_test_file_path(DIR1_REL);
  auto dir2 = libc_make_test_file_path(DIR2_REL);
  auto file3 = libc_make_test_file_path(FILE3_REL);
  auto dirb = libc_make_test_file_path(DIRB_REL);
  auto fileb = libc_make_test_file_path(FILEB_REL);

  ASSERT_THAT(LIBC_NAMESPACE::mkdir(root, 0755), Succeeds(0));
  auto cleanup = LIBC_NAMESPACE::cpp::scope_exit([&]() {
    LIBC_NAMESPACE::unlink(fileb);
    LIBC_NAMESPACE::rmdir(dirb);
    LIBC_NAMESPACE::unlink(file3);
    LIBC_NAMESPACE::rmdir(dir2);
    LIBC_NAMESPACE::rmdir(dir1);
    LIBC_NAMESPACE::rmdir(root);
  });

  ASSERT_THAT(LIBC_NAMESPACE::mkdir(dir1, 0755), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::mkdir(dir2, 0755), Succeeds(0));
  ASSERT_THAT(create_empty_file(file3), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::mkdir(dirb, 0755), Succeeds(0));
  ASSERT_THAT(create_empty_file(fileb), Succeeds(0));

  char cwd_before[4096];
  char cwd_after[4096];
  ASSERT_THAT(LIBC_NAMESPACE::getcwd(cwd_before, sizeof(cwd_before)),
              Succeeds(cwd_before));
  auto restore_cwd = LIBC_NAMESPACE::cpp::scope_exit(
      [&]() { LIBC_NAMESPACE::chdir(cwd_before); });

  // Exercise fd_limit across buffered (1), mixed transition (2, 3), and
  // keep-open (10) modes, walking a root path with a slash (DIR1_REL) so
  // level-0 FTW_CHDIR must chdir into dirname(DIR1_REL) before calling fn.
  constexpr int LIMITS[] = {1, 2, 3, 10};
  for (int limit : LIMITS) {
    reset_counts();
    ASSERT_THAT(LIBC_NAMESPACE::nftw(root, record_callback, limit, FTW_PHYS),
                Succeeds(0));
    EXPECT_EQ(call_count, 6);
    EXPECT_EQ(dir_count, 4);
    EXPECT_EQ(file_count, 2);
    EXPECT_EQ(max_level, 3);

    reset_counts();
    ASSERT_THAT(
        LIBC_NAMESPACE::nftw(dir1, chdir_callback, limit, FTW_CHDIR | FTW_PHYS),
        Succeeds(0));
    ASSERT_THAT(LIBC_NAMESPACE::getcwd(cwd_after, sizeof(cwd_after)),
                Succeeds(cwd_after));
    EXPECT_STREQ(cwd_before, cwd_after);
    EXPECT_EQ(call_count, 5);

    reset_counts();
    ASSERT_THAT(LIBC_NAMESPACE::nftw(dir1, chdir_callback, limit,
                                     FTW_CHDIR | FTW_PHYS | FTW_DEPTH),
                Succeeds(0));
    ASSERT_THAT(LIBC_NAMESPACE::getcwd(cwd_after, sizeof(cwd_after)),
                Succeeds(cwd_after));
    EXPECT_STREQ(cwd_before, cwd_after);
    EXPECT_EQ(call_count, 5);
  }

  // Early callback stop at level == 1 with FTW_CHDIR and fd_limit == 1 must
  // still restore the caller's working directory.
  reset_counts();
  ASSERT_THAT(
      LIBC_NAMESPACE::nftw(dir1, stop_at_level1_cb, 1, FTW_CHDIR | FTW_PHYS),
      Succeeds(99));
  ASSERT_THAT(LIBC_NAMESPACE::getcwd(cwd_after, sizeof(cwd_after)),
              Succeeds(cwd_after));
  EXPECT_STREQ(cwd_before, cwd_after);
  EXPECT_EQ(call_count, 2);
}

TEST_F(LlvmLibcNftwTest, BufferGrowthAndVanishedEntry) {
  // Exercise PathBuffer growth (>256 bytes) and DirEntryNames growth (>256
  // bytes under fd_limit == 1), plus ENOENT->FTW_NS when a sibling vanishes
  // mid-walk.
  constexpr const char *RACE_ROOT = "nftw_race_root";
  constexpr const char *VANISH_A = "nftw_race_root/vanish_a.txt";
  constexpr const char *VANISH_B = "nftw_race_root/vanish_b.txt";
  constexpr const char *LONG_A =
      "nftw_race_root/"
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
  constexpr const char *LONG_B =
      "nftw_race_root/"
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/"
      "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
      "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
  constexpr const char *LONG_FILE =
      "nftw_race_root/"
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/"
      "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
      "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb/"
      "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
      "cccccccccccccccccccccccccccccccccccccccccccccccccccccccc.txt";

  auto cleanup = LIBC_NAMESPACE::cpp::scope_exit([&]() {
    LIBC_NAMESPACE::unlink(LONG_FILE);
    LIBC_NAMESPACE::rmdir(LONG_B);
    LIBC_NAMESPACE::rmdir(LONG_A);
    LIBC_NAMESPACE::unlink(VANISH_A);
    LIBC_NAMESPACE::unlink(VANISH_B);
    LIBC_NAMESPACE::rmdir(RACE_ROOT);
    LIBC_NAMESPACE::libc_errno = 0;
  });

  ASSERT_THAT(LIBC_NAMESPACE::mkdir(RACE_ROOT, 0755), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::mkdir(LONG_A, 0755), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::mkdir(LONG_B, 0755), Succeeds(0));
  ASSERT_THAT(create_empty_file(LONG_FILE), Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::nftw(RACE_ROOT, record_callback, 1, FTW_PHYS),
              Succeeds(0));
  EXPECT_EQ(call_count, 4);
  EXPECT_EQ(dir_count, 3);
  EXPECT_EQ(file_count, 1);

  // When one sibling unlinks the other during iteration, the vanished sibling
  // is reported as FTW_NS and the walk continues rather than aborting.
  ASSERT_THAT(create_empty_file(VANISH_A), Succeeds(0));
  ASSERT_THAT(create_empty_file(VANISH_B), Succeeds(0));
  reset_counts();
  ASSERT_THAT(LIBC_NAMESPACE::nftw(RACE_ROOT, unlink_sibling_cb, 1, FTW_PHYS),
              Succeeds(0));
  EXPECT_EQ(call_count, 6);
  EXPECT_EQ(ns_count, 1);
}

TEST_F(LlvmLibcNftwTest, CallbackReturnAndErrorCases) {
  constexpr const char *ROOT_REL = "nftw_err_root";
  constexpr const char *FILE_REL = "nftw_err_root/file.txt";
  constexpr const char *FILE_SLASH_REL = "nftw_err_root/file.txt/";
  constexpr const char *NOTDIR_REL = "nftw_err_root/file.txt/child";
  constexpr const char *NONEXISTENT_REL = "nftw_nonexistent_path";

  auto root = libc_make_test_file_path(ROOT_REL);
  auto file_path = libc_make_test_file_path(FILE_REL);
  auto file_slash_path = libc_make_test_file_path(FILE_SLASH_REL);
  auto notdir_path = libc_make_test_file_path(NOTDIR_REL);
  auto nonexistent = libc_make_test_file_path(NONEXISTENT_REL);

  ASSERT_THAT(LIBC_NAMESPACE::mkdir(root, 0755), Succeeds(0));
  auto cleanup = LIBC_NAMESPACE::cpp::scope_exit([&]() {
    LIBC_NAMESPACE::unlink(file_path);
    LIBC_NAMESPACE::rmdir(root);
  });
  ASSERT_THAT(create_empty_file(file_path), Succeeds(0));

  // Passing a regular file directly as the root path reports FTW_F at level 0.
  ASSERT_THAT(LIBC_NAMESPACE::nftw(file_path, record_callback, 10, 0),
              Succeeds(0));
  EXPECT_EQ(call_count, 1);
  EXPECT_EQ(file_count, 1);
  EXPECT_EQ(max_level, 0);

  reset_counts();
  ASSERT_THAT(LIBC_NAMESPACE::nftw(root, stop_42_cb, 10, 0), Succeeds(42));
  EXPECT_EQ(call_count, 1);

  // Returning -1 from callback must preserve the errno set by the callback.
  reset_counts();
  ASSERT_THAT(LIBC_NAMESPACE::nftw(root, stop_neg_one_cb, 10, 0), Fails(EXDEV));
  EXPECT_EQ(call_count, 1);

  EXPECT_THAT(LIBC_NAMESPACE::nftw("", record_callback, 10, 0), Fails(ENOENT));
  EXPECT_THAT(LIBC_NAMESPACE::nftw(nonexistent, record_callback, 10, 0),
              Fails(ENOENT));
  EXPECT_THAT(LIBC_NAMESPACE::nftw(file_slash_path, record_callback, 10, 0),
              Fails(ENOTDIR));
  EXPECT_THAT(LIBC_NAMESPACE::nftw(notdir_path, record_callback, 10, 0),
              Fails(ENOTDIR));
  EXPECT_THAT(LIBC_NAMESPACE::nftw(root, record_callback, 0, 0), Fails(EINVAL));
  EXPECT_THAT(LIBC_NAMESPACE::nftw(root, record_callback, -1, 0),
              Fails(EINVAL));
  EXPECT_THAT(LIBC_NAMESPACE::nftw(root, record_callback, 10, ~0),
              Fails(EINVAL));
}

TEST_F(LlvmLibcNftwTest, PermissionErrors) {
  if (LIBC_NAMESPACE::geteuid() == 0) {
    tlog << "Skipping PermissionErrors: running as root.\n";
    return;
  }

  constexpr const char *ROOT_REL = "nftw_perm_root";
  constexpr const char *UNREADABLE_REL = "nftw_perm_root/unreadable_dir";
  constexpr const char *NOSEARCH_REL = "nftw_perm_root/nosearch_dir";
  constexpr const char *CHILD_REL = "nftw_perm_root/nosearch_dir/child";

  auto root = libc_make_test_file_path(ROOT_REL);
  auto unreadable = libc_make_test_file_path(UNREADABLE_REL);
  auto nosearch = libc_make_test_file_path(NOSEARCH_REL);
  auto child = libc_make_test_file_path(CHILD_REL);

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
  ASSERT_THAT(create_empty_file(child), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::chmod(nosearch, 0666), Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::nftw(root, record_callback, 10, 0), Succeeds(0));
  EXPECT_EQ(call_count, 4);
  EXPECT_EQ(dir_count, 2);
  EXPECT_EQ(dnr_count, 1);
  EXPECT_EQ(ns_count, 1);

  EXPECT_THAT(LIBC_NAMESPACE::nftw(unreadable, record_callback, 10, 0),
              Fails(EACCES));
  EXPECT_THAT(LIBC_NAMESPACE::nftw(child, record_callback, 10, 0),
              Fails(EACCES));
}

TEST_F(LlvmLibcNftwTest, FlagsAndActionRetVal) {
  constexpr const char *ROOT_REL = "nftw_flags_root";
  constexpr const char *SUBDIR_REL = "nftw_flags_root/dir1";
  constexpr const char *FILE1_REL = "nftw_flags_root/file1.txt";
  constexpr const char *FILE2_REL = "nftw_flags_root/dir1/file2.txt";
  constexpr const char *SYMLINK_REL = "nftw_flags_root/link_to_file1";
  constexpr const char *BROKEN_REL = "nftw_flags_root/broken_link";
  constexpr const char *BROKEN_SLASH_REL = "nftw_flags_root/broken_link/";

  auto root = libc_make_test_file_path(ROOT_REL);
  auto subdir = libc_make_test_file_path(SUBDIR_REL);
  auto file1 = libc_make_test_file_path(FILE1_REL);
  auto file2 = libc_make_test_file_path(FILE2_REL);
  auto symlink_path = libc_make_test_file_path(SYMLINK_REL);
  auto broken_path = libc_make_test_file_path(BROKEN_REL);
  auto broken_slash_path = libc_make_test_file_path(BROKEN_SLASH_REL);

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
  ASSERT_THAT(create_empty_file(file1), Succeeds(0));
  ASSERT_THAT(create_empty_file(file2), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::symlink("file1.txt", symlink_path), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::symlink("nonexistent_target", broken_path),
              Succeeds(0));

  // Trailing slash on a dangling symlink forces target resolution and fails
  // with ENOENT rather than reporting FTW_SL or FTW_SLN.
  EXPECT_THAT(LIBC_NAMESPACE::nftw(broken_slash_path, record_callback, 10, 0),
              Fails(ENOENT));
  EXPECT_THAT(
      LIBC_NAMESPACE::nftw(broken_slash_path, record_callback, 10, FTW_PHYS),
      Fails(ENOENT));

  ASSERT_THAT(LIBC_NAMESPACE::nftw(root, record_callback, 10, FTW_MOUNT),
              Succeeds(0));
  EXPECT_EQ(max_level, 2);
  EXPECT_EQ(dir_count, 2);
  EXPECT_EQ(file_count, 3);
  EXPECT_EQ(sln_count, 1);
  EXPECT_EQ(sl_count, 0);

  // FTW_PHYS + FTW_DEPTH: both symlinks are FTW_SL, directories are FTW_DP,
  // and subdir is visited before root.
  reset_counts();
  ASSERT_THAT(
      LIBC_NAMESPACE::nftw(root, record_callback, 10, FTW_PHYS | FTW_DEPTH),
      Succeeds(0));
  EXPECT_EQ(dp_count, 2);
  EXPECT_EQ(sl_count, 2);
  EXPECT_GT(find_visit(ROOT_REL), find_visit(SUBDIR_REL));
  EXPECT_GT(find_visit(SUBDIR_REL), find_visit(FILE2_REL));

  constexpr int LIMITS[] = {1, 10};
  for (int limit : LIMITS) {
    // FTW_ACTIONRETVAL: FTW_SKIP_SUBTREE skips file2.txt inside dir1 (6 - 1 =
    // 5).
    reset_counts();
    ASSERT_THAT(LIBC_NAMESPACE::nftw(root, skip_subtree_cb, limit,
                                     FTW_PHYS | FTW_ACTIONRETVAL),
                Succeeds(0));
    EXPECT_EQ(call_count, 5);

    // FTW_ACTIONRETVAL: FTW_SKIP_SIBLINGS at level 1 stops after the first
    // level-1 entry and returns to parent (root + first child = 2).
    reset_counts();
    ASSERT_THAT(LIBC_NAMESPACE::nftw(root, skip_siblings_cb, limit,
                                     FTW_PHYS | FTW_ACTIONRETVAL),
                Succeeds(0));
    EXPECT_EQ(call_count, 2);

    // FTW_ACTIONRETVAL: FTW_STOP halts immediately and returns FTW_STOP.
    reset_counts();
    ASSERT_THAT(
        LIBC_NAMESPACE::nftw(root, stop_action_cb, limit, FTW_ACTIONRETVAL),
        Succeeds<int>(FTW_STOP));
    EXPECT_EQ(call_count, 1);
  }
}

#if defined(LIBC_ADD_NULL_CHECKS)
TEST_F(LlvmLibcNftwTest, NullptrCrash) {
  ASSERT_DEATH([] { LIBC_NAMESPACE::nftw(nullptr, record_callback, 10, 0); },
               WITH_SIGNAL(-1));
  ASSERT_DEATH([] { LIBC_NAMESPACE::nftw(".", nullptr, 10, 0); },
               WITH_SIGNAL(-1));
}
#endif // LIBC_ADD_NULL_CHECKS
