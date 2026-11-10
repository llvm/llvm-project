//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tests for mkostemp
/// See: https://pubs.opengroup.org/onlinepubs/9799919799/functions/mkdtemp.html
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "hdr/func/free.h"
#include "hdr/signal_macros.h"
#include "hdr/stdio_macros.h"
#include "hdr/sys_stat_macros.h"
#include "hdr/types/struct_stat.h"
#include "hdr/unistd_macros.h"
#include "src/__support/CPP/scope.h"
#include "src/__support/CPP/string_view.h"
#include "src/fcntl/fcntl.h"
#include "src/stdlib/mkostemp.h"
#include "src/string/strdup.h"
#include "src/string/strlen.h"
#include "src/sys/stat/stat.h"
#include "src/unistd/access.h"
#include "src/unistd/close.h"
#include "src/unistd/lseek.h"
#include "src/unistd/read.h"
#include "src/unistd/unlink.h"
#include "src/unistd/write.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/MemoryMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LIBC_NAMESPACE::cpp::string_view;
using LIBC_NAMESPACE::testing::MemoryView;
using LlvmLibcMkostempTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcMkostempTest, ValidTemplateDefaultFlags) {
  char *tmpl = LIBC_NAMESPACE::strdup(libc_make_test_file_path("tmp_XXXXXX"));
  ASSERT_NE(tmpl, nullptr);
  auto cleanup_tmpl = LIBC_NAMESPACE::cpp::scope_exit([&] {
    LIBC_NAMESPACE::unlink(tmpl);
    ::free(tmpl);
  });

  int fd = -1;
  ASSERT_THAT(fd = LIBC_NAMESPACE::mkostemp(tmpl, 0),
              returns(GE(0)).with_errno(EQ(0)));
  auto cleanup_fd =
      LIBC_NAMESPACE::cpp::scope_exit([&] { LIBC_NAMESPACE::close(fd); });

  EXPECT_THAT(LIBC_NAMESPACE::access(tmpl, F_OK), Succeeds(0));

  struct stat st;
  ASSERT_THAT(LIBC_NAMESPACE::stat(tmpl, &st), Succeeds(0));
  EXPECT_EQ(st.st_mode & S_IFMT, static_cast<mode_t>(S_IFREG));
  EXPECT_EQ(st.st_mode & (S_IRWXU | S_IRWXG | S_IRWXO),
            static_cast<mode_t>(S_IRUSR | S_IWUSR));

  ASSERT_THAT(LIBC_NAMESPACE::write(fd, "llvm", 4),
              Succeeds(static_cast<ssize_t>(4)));
}

TEST_F(LlvmLibcMkostempTest, TemplateModifiedInPlace) {
  char *tmpl = LIBC_NAMESPACE::strdup(libc_make_test_file_path("tmp_XXXXXX"));
  ASSERT_NE(tmpl, nullptr);
  auto cleanup_tmpl = LIBC_NAMESPACE::cpp::scope_exit([&] {
    LIBC_NAMESPACE::unlink(tmpl);
    ::free(tmpl);
  });

  char *orig = LIBC_NAMESPACE::strdup(tmpl);
  ASSERT_NE(orig, nullptr);
  auto cleanup_orig = LIBC_NAMESPACE::cpp::scope_exit([&] { ::free(orig); });

  size_t len = LIBC_NAMESPACE::strlen(tmpl);
  int fd = -1;
  ASSERT_THAT(fd = LIBC_NAMESPACE::mkostemp(tmpl, 0),
              returns(GE(0)).with_errno(EQ(0)));
  auto cleanup_fd =
      LIBC_NAMESPACE::cpp::scope_exit([&] { LIBC_NAMESPACE::close(fd); });

  EXPECT_EQ(string_view(tmpl, len - 6), string_view(orig, len - 6));
  EXPECT_NE(string_view(tmpl, len).substr(len - 6), string_view("XXXXXX"));
}

TEST_F(LlvmLibcMkostempTest, AllCharactersInCharset) {
  char *tmpl = LIBC_NAMESPACE::strdup(libc_make_test_file_path("tmp_XXXXXX"));
  ASSERT_NE(tmpl, nullptr);
  auto cleanup_tmpl = LIBC_NAMESPACE::cpp::scope_exit([&] {
    LIBC_NAMESPACE::unlink(tmpl);
    ::free(tmpl);
  });

  size_t len = LIBC_NAMESPACE::strlen(tmpl);
  int fd = -1;
  ASSERT_THAT(fd = LIBC_NAMESPACE::mkostemp(tmpl, 0),
              returns(GE(0)).with_errno(EQ(0)));
  auto cleanup_fd =
      LIBC_NAMESPACE::cpp::scope_exit([&] { LIBC_NAMESPACE::close(fd); });

  // POSIX portable filename character set, sorted by ASCII value.
  // See
  // https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/V1_chap03.html#tag_03_265
  constexpr string_view CHARSET = "-._0123456789"
                                  "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                  "abcdefghijklmnopqrstuvwxyz";
  for (char c : string_view(tmpl, len).substr(len - 6))
    EXPECT_NE(CHARSET.find_first_of(c), string_view::npos);
}

TEST_F(LlvmLibcMkostempTest, Uniqueness) {
  char *tmpl1 = LIBC_NAMESPACE::strdup(libc_make_test_file_path("tmp_XXXXXX"));
  ASSERT_NE(tmpl1, nullptr);
  auto cleanup_tmpl1 = LIBC_NAMESPACE::cpp::scope_exit([&] {
    LIBC_NAMESPACE::unlink(tmpl1);
    ::free(tmpl1);
  });

  char *tmpl2 = LIBC_NAMESPACE::strdup(libc_make_test_file_path("tmp_XXXXXX"));
  ASSERT_NE(tmpl2, nullptr);
  auto cleanup_tmpl2 = LIBC_NAMESPACE::cpp::scope_exit([&] {
    LIBC_NAMESPACE::unlink(tmpl2);
    ::free(tmpl2);
  });

  int fd1 = -1;
  ASSERT_THAT(fd1 = LIBC_NAMESPACE::mkostemp(tmpl1, 0),
              returns(GE(0)).with_errno(EQ(0)));
  auto cleanup_fd1 =
      LIBC_NAMESPACE::cpp::scope_exit([&] { LIBC_NAMESPACE::close(fd1); });

  int fd2 = -1;
  ASSERT_THAT(fd2 = LIBC_NAMESPACE::mkostemp(tmpl2, 0),
              returns(GE(0)).with_errno(EQ(0)));
  auto cleanup_fd2 =
      LIBC_NAMESPACE::cpp::scope_exit([&] { LIBC_NAMESPACE::close(fd2); });

  EXPECT_STRNE(tmpl1, tmpl2);
}

TEST_F(LlvmLibcMkostempTest, FlagCloexec) {
  char *tmpl = LIBC_NAMESPACE::strdup(libc_make_test_file_path("tmp_XXXXXX"));
  ASSERT_NE(tmpl, nullptr);
  auto cleanup_tmpl = LIBC_NAMESPACE::cpp::scope_exit([&] {
    LIBC_NAMESPACE::unlink(tmpl);
    ::free(tmpl);
  });

  int fd = -1;
  ASSERT_THAT(fd = LIBC_NAMESPACE::mkostemp(tmpl, O_CLOEXEC),
              returns(GE(0)).with_errno(EQ(0)));
  auto cleanup_fd =
      LIBC_NAMESPACE::cpp::scope_exit([&] { LIBC_NAMESPACE::close(fd); });

  int fd_flags = LIBC_NAMESPACE::fcntl(fd, F_GETFD);
  ASSERT_GE(fd_flags, 0);
  EXPECT_EQ(fd_flags & FD_CLOEXEC, FD_CLOEXEC);
}

TEST_F(LlvmLibcMkostempTest, FlagAppend) {
  char *tmpl = LIBC_NAMESPACE::strdup(libc_make_test_file_path("tmp_XXXXXX"));
  ASSERT_NE(tmpl, nullptr);
  auto cleanup_tmpl = LIBC_NAMESPACE::cpp::scope_exit([&] {
    LIBC_NAMESPACE::unlink(tmpl);
    ::free(tmpl);
  });

  int fd = -1;
  ASSERT_THAT(fd = LIBC_NAMESPACE::mkostemp(tmpl, O_APPEND),
              returns(GE(0)).with_errno(EQ(0)));
  auto cleanup_fd =
      LIBC_NAMESPACE::cpp::scope_exit([&] { LIBC_NAMESPACE::close(fd); });

  int fl_flags = LIBC_NAMESPACE::fcntl(fd, F_GETFL);
  ASSERT_GE(fl_flags, 0);
  EXPECT_EQ(fl_flags & O_APPEND, O_APPEND);

  ASSERT_THAT(LIBC_NAMESPACE::write(fd, "abc", 3),
              Succeeds(static_cast<ssize_t>(3)));
  ASSERT_THAT(LIBC_NAMESPACE::lseek(fd, 0, SEEK_SET),
              Succeeds(static_cast<off_t>(0)));
  // With O_APPEND, writes must always append to the end of the file.
  ASSERT_THAT(LIBC_NAMESPACE::write(fd, "def", 3),
              Succeeds(static_cast<ssize_t>(3)));
  ASSERT_THAT(LIBC_NAMESPACE::lseek(fd, 0, SEEK_SET),
              Succeeds(static_cast<off_t>(0)));

  char buf[6];
  ASSERT_THAT(LIBC_NAMESPACE::read(fd, buf, 6),
              Succeeds(static_cast<ssize_t>(6)));
  EXPECT_MEM_EQ(MemoryView("abcdef", 6), MemoryView(buf, 6));
}

TEST_F(LlvmLibcMkostempTest, FlagSyncVariants) {
  constexpr int SYNC_FLAGS[] = {O_SYNC, O_DSYNC};
  for (int sync_flag : SYNC_FLAGS) {
    char *tmpl = LIBC_NAMESPACE::strdup(libc_make_test_file_path("tmp_XXXXXX"));
    ASSERT_NE(tmpl, nullptr);
    auto cleanup_tmpl = LIBC_NAMESPACE::cpp::scope_exit([&] {
      LIBC_NAMESPACE::unlink(tmpl);
      ::free(tmpl);
    });

    int fd = -1;
    ASSERT_THAT(fd = LIBC_NAMESPACE::mkostemp(tmpl, sync_flag),
                returns(GE(0)).with_errno(EQ(0)));
    auto cleanup_fd =
        LIBC_NAMESPACE::cpp::scope_exit([&] { LIBC_NAMESPACE::close(fd); });
    int fl = LIBC_NAMESPACE::fcntl(fd, F_GETFL);
    ASSERT_GE(fl, 0);
    EXPECT_EQ(fl & sync_flag, sync_flag);
  }
}

TEST_F(LlvmLibcMkostempTest, CombinedFlags) {
  char *tmpl = LIBC_NAMESPACE::strdup(libc_make_test_file_path("tmp_XXXXXX"));
  ASSERT_NE(tmpl, nullptr);
  auto cleanup_tmpl = LIBC_NAMESPACE::cpp::scope_exit([&] {
    LIBC_NAMESPACE::unlink(tmpl);
    ::free(tmpl);
  });

  int fd = -1;
  ASSERT_THAT(fd = LIBC_NAMESPACE::mkostemp(tmpl, O_CLOEXEC | O_APPEND),
              returns(GE(0)).with_errno(EQ(0)));
  auto cleanup_fd =
      LIBC_NAMESPACE::cpp::scope_exit([&] { LIBC_NAMESPACE::close(fd); });

  int fd_flags = LIBC_NAMESPACE::fcntl(fd, F_GETFD);
  ASSERT_GE(fd_flags, 0);
  EXPECT_EQ(fd_flags & FD_CLOEXEC, FD_CLOEXEC);

  int fl_flags = LIBC_NAMESPACE::fcntl(fd, F_GETFL);
  ASSERT_GE(fl_flags, 0);
  EXPECT_EQ(fl_flags & O_APPEND, O_APPEND);
}

TEST_F(LlvmLibcMkostempTest, SixXsNoPrefix) {
  char *tmpl = LIBC_NAMESPACE::strdup(libc_make_test_file_path("XXXXXX"));
  ASSERT_NE(tmpl, nullptr);
  auto cleanup_tmpl = LIBC_NAMESPACE::cpp::scope_exit([&] {
    LIBC_NAMESPACE::unlink(tmpl);
    ::free(tmpl);
  });

  int fd = -1;
  ASSERT_THAT(fd = LIBC_NAMESPACE::mkostemp(tmpl, 0),
              returns(GE(0)).with_errno(EQ(0)));
  auto cleanup_fd =
      LIBC_NAMESPACE::cpp::scope_exit([&] { LIBC_NAMESPACE::close(fd); });

  EXPECT_THAT(LIBC_NAMESPACE::access(tmpl, F_OK), Succeeds(0));
}

TEST_F(LlvmLibcMkostempTest, MoreThanSixXs) {
  char *tmpl =
      LIBC_NAMESPACE::strdup(libc_make_test_file_path("tmp_XXXXXXXXXX"));
  ASSERT_NE(tmpl, nullptr);
  auto cleanup_tmpl = LIBC_NAMESPACE::cpp::scope_exit([&] {
    LIBC_NAMESPACE::unlink(tmpl);
    ::free(tmpl);
  });

  char *orig = LIBC_NAMESPACE::strdup(tmpl);
  ASSERT_NE(orig, nullptr);
  auto cleanup_orig = LIBC_NAMESPACE::cpp::scope_exit([&] { ::free(orig); });

  size_t len = LIBC_NAMESPACE::strlen(tmpl);
  int fd = -1;
  ASSERT_THAT(fd = LIBC_NAMESPACE::mkostemp(tmpl, 0),
              returns(GE(0)).with_errno(EQ(0)));
  auto cleanup_fd =
      LIBC_NAMESPACE::cpp::scope_exit([&] { LIBC_NAMESPACE::close(fd); });

  EXPECT_EQ(string_view(tmpl, len - 10), string_view(orig, len - 10));
  EXPECT_NE(string_view(tmpl, len).substr(len - 10), string_view("XXXXXXXXXX"));
  EXPECT_THAT(LIBC_NAMESPACE::access(tmpl, F_OK), Succeeds(0));
}

#if defined(LIBC_ADD_NULL_CHECKS)
TEST_F(LlvmLibcMkostempTest, NullPointer) {
  ASSERT_DEATH([] { LIBC_NAMESPACE::mkostemp(nullptr, 0); }, WITH_SIGNAL(-1));
}
#endif

TEST_F(LlvmLibcMkostempTest, InvalidFlags) {
  char tmpl[] = "tmp_XXXXXX";
  EXPECT_THAT(LIBC_NAMESPACE::mkostemp(tmpl, O_CREAT), Fails(EINVAL));
  EXPECT_THAT(LIBC_NAMESPACE::mkostemp(tmpl, O_EXCL), Fails(EINVAL));
  EXPECT_THAT(LIBC_NAMESPACE::mkostemp(tmpl, O_RDWR), Fails(EINVAL));
  EXPECT_THAT(LIBC_NAMESPACE::mkostemp(tmpl, O_WRONLY), Fails(EINVAL));
  EXPECT_THAT(LIBC_NAMESPACE::mkostemp(tmpl, O_TRUNC), Fails(EINVAL));
  EXPECT_THAT(LIBC_NAMESPACE::mkostemp(tmpl, -1), Fails(EINVAL));
  EXPECT_STREQ(tmpl, "tmp_XXXXXX");
}

TEST_F(LlvmLibcMkostempTest, TemplateTooShort) {
  char tmpl[] = "XXXXX";
  EXPECT_THAT(LIBC_NAMESPACE::mkostemp(tmpl, 0), Fails(EINVAL));
}

TEST_F(LlvmLibcMkostempTest, DoesNotEndInXs) {
  char tmpl[] = "tmp_XXXXXY";
  EXPECT_THAT(LIBC_NAMESPACE::mkostemp(tmpl, 0), Fails(EINVAL));
}

TEST_F(LlvmLibcMkostempTest, XsNotAtEnd) {
  char tmpl[] = "XXXXXXtmp";
  EXPECT_THAT(LIBC_NAMESPACE::mkostemp(tmpl, 0), Fails(EINVAL));
}

TEST_F(LlvmLibcMkostempTest, FiveXsAtEnd) {
  char tmpl[] = "tmp_XXXXX";
  EXPECT_THAT(LIBC_NAMESPACE::mkostemp(tmpl, 0), Fails(EINVAL));
}

TEST_F(LlvmLibcMkostempTest, EmptyString) {
  char tmpl[] = "";
  EXPECT_THAT(LIBC_NAMESPACE::mkostemp(tmpl, 0), Fails(EINVAL));
}

TEST_F(LlvmLibcMkostempTest, NonExistentParentDirectory) {
  char *tmpl = LIBC_NAMESPACE::strdup(
      libc_make_test_file_path("non_existent_dir/tmp_XXXXXX"));
  ASSERT_NE(tmpl, nullptr);
  auto cleanup = LIBC_NAMESPACE::cpp::scope_exit([&] { ::free(tmpl); });
  EXPECT_THAT(LIBC_NAMESPACE::mkostemp(tmpl, 0), Fails(ENOENT));
}
