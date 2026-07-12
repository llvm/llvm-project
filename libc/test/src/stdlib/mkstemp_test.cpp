//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tests for mkstemp
/// See: https://pubs.opengroup.org/onlinepubs/9799919799/functions/mkdtemp.html
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "hdr/signal_macros.h"
#include "src/stdlib/mkstemp.h"
#include "src/string/strdup.h"
#include "src/string/strlen.h"
#include "src/unistd/access.h"
#include "src/unistd/close.h"
#include "src/unistd/read.h"
#include "src/unistd/unlink.h"
#include "src/unistd/write.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcMkstempTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcMkstempTest, ValidTemplate) {
  char *tmpl = LIBC_NAMESPACE::strdup(libc_make_test_file_path("tmp_XXXXXX"));
  int fd = LIBC_NAMESPACE::mkstemp(tmpl);
  ASSERT_GE(fd, 0);
  LIBC_NAMESPACE::close(fd);
  LIBC_NAMESPACE::unlink(tmpl);
  ::free(tmpl);
}

TEST_F(LlvmLibcMkstempTest, TemplateModifiedInPlace) {
  char *tmpl = LIBC_NAMESPACE::strdup(libc_make_test_file_path("tmp_XXXXXX"));
  size_t len = LIBC_NAMESPACE::strlen(tmpl);
  size_t count = 0;
  for (size_t i = len; i > 0 && tmpl[i - 1] == 'X'; i--)
    count++;
  int fd = LIBC_NAMESPACE::mkstemp(tmpl);
  ASSERT_GE(fd, 0);
  bool modified = false;
  for (size_t i = len - count; i < len; i++)
    if (tmpl[i] != 'X') {
      modified = true;
      break;
    }
  EXPECT_TRUE(modified);
  LIBC_NAMESPACE::close(fd);
  LIBC_NAMESPACE::unlink(tmpl);
  ::free(tmpl);
}

TEST_F(LlvmLibcMkstempTest, AllCharactersInCharset) {
  char *tmpl = LIBC_NAMESPACE::strdup(libc_make_test_file_path("tmp_XXXXXX"));
  size_t len = LIBC_NAMESPACE::strlen(tmpl);
  size_t count = 0;
  for (size_t i = len; i > 0 && tmpl[i - 1] == 'X'; i--)
    count++;
  int fd = LIBC_NAMESPACE::mkstemp(tmpl);
  ASSERT_GE(fd, 0);
  // POSIX portable filename character set, sorted by ASCII value.
  // See
  // https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/V1_chap03.html#tag_03_265
  const char charset[] = "-._0123456789"
                         "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                         "abcdefghijklmnopqrstuvwxyz";
  for (size_t i = len - count; i < len; i++) {
    bool found = false;
    for (size_t j = 0; j < sizeof(charset) - 1; j++) {
      if (tmpl[i] == charset[j]) {
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found);
  }
  LIBC_NAMESPACE::close(fd);
  LIBC_NAMESPACE::unlink(tmpl);
  ::free(tmpl);
}

TEST_F(LlvmLibcMkstempTest, FileOpenedCorrectly) {
  char *tmpl = LIBC_NAMESPACE::strdup(libc_make_test_file_path("tmp_XXXXXX"));
  int fd = LIBC_NAMESPACE::mkstemp(tmpl);
  ASSERT_GE(fd, 0);
  // Check for two things:
  // 1. The file exists on the disk
  EXPECT_EQ(LIBC_NAMESPACE::access(tmpl, F_OK), 0);
  // 2. The file can be written to
  EXPECT_EQ(LIBC_NAMESPACE::write(fd, "llvm", 4), static_cast<ssize_t>(4));
  LIBC_NAMESPACE::close(fd);
  LIBC_NAMESPACE::unlink(tmpl);
  ::free(tmpl);
}

TEST_F(LlvmLibcMkstempTest, Uniqueness) {
  char *tmpl1 = LIBC_NAMESPACE::strdup(libc_make_test_file_path("tmp_XXXXXX"));
  char *tmpl2 = LIBC_NAMESPACE::strdup(libc_make_test_file_path("tmp_XXXXXX"));
  int fd1 = LIBC_NAMESPACE::mkstemp(tmpl1);
  int fd2 = LIBC_NAMESPACE::mkstemp(tmpl2);
  ASSERT_GE(fd1, 0);
  ASSERT_GE(fd2, 0);
  bool different = false;
  for (size_t i = 0; tmpl1[i] != '\0'; i++)
    if (tmpl1[i] != tmpl2[i]) {
      different = true;
      break;
    }
  EXPECT_TRUE(different);
  LIBC_NAMESPACE::close(fd1);
  LIBC_NAMESPACE::close(fd2);
  LIBC_NAMESPACE::unlink(tmpl1);
  LIBC_NAMESPACE::unlink(tmpl2);
  ::free(tmpl1);
  ::free(tmpl2);
}

TEST_F(LlvmLibcMkstempTest, FileIsEmpty) {
  char *tmpl = LIBC_NAMESPACE::strdup(libc_make_test_file_path("tmp_XXXXXX"));
  int fd = LIBC_NAMESPACE::mkstemp(tmpl);
  ASSERT_GE(fd, 0);
  // read should return 0 on empty file
  char buf[1];
  EXPECT_EQ(LIBC_NAMESPACE::read(fd, buf, 1), static_cast<ssize_t>(0));
  LIBC_NAMESPACE::close(fd);
  LIBC_NAMESPACE::unlink(tmpl);
  ::free(tmpl);
}

TEST_F(LlvmLibcMkstempTest, SixXsNoPrefix) {
  char *tmpl = LIBC_NAMESPACE::strdup(libc_make_test_file_path("XXXXXX"));
  int fd = LIBC_NAMESPACE::mkstemp(tmpl);
  ASSERT_GE(fd, 0);
  LIBC_NAMESPACE::close(fd);
  LIBC_NAMESPACE::unlink(tmpl);
  ::free(tmpl);
}

TEST_F(LlvmLibcMkstempTest, MoreThanSixXs) {
  char *tmpl =
      LIBC_NAMESPACE::strdup(libc_make_test_file_path("tmp_XXXXXXXXXX"));
  size_t len = LIBC_NAMESPACE::strlen(tmpl);
  size_t count = 0;
  for (size_t i = len; i > 0 && tmpl[i - 1] == 'X'; i--)
    count++;
  int fd = LIBC_NAMESPACE::mkstemp(tmpl);
  ASSERT_GE(fd, 0);
  bool modified = false;
  for (size_t i = len - count; i < len; i++)
    if (tmpl[i] != 'X') {
      modified = true;
      break;
    }
  EXPECT_TRUE(modified);
  EXPECT_EQ(LIBC_NAMESPACE::access(tmpl, F_OK), 0);
  LIBC_NAMESPACE::close(fd);
  LIBC_NAMESPACE::unlink(tmpl);
  ::free(tmpl);
}

#if defined(LIBC_ADD_NULL_CHECKS)
TEST_F(LlvmLibcMkstempTest, NullPointer) {
  EXPECT_DEATH([] { LIBC_NAMESPACE::mkstemp(nullptr); }, WITH_SIGNAL(-1));
}
#endif

TEST_F(LlvmLibcMkstempTest, TemplateTooShort) {
  char tmpl[] = "XXXXX";
  int fd = LIBC_NAMESPACE::mkstemp(tmpl);
  EXPECT_EQ(fd, -1);
  ASSERT_ERRNO_EQ(EINVAL);
}

TEST_F(LlvmLibcMkstempTest, DoesNotEndInXs) {
  char tmpl[] = "tmp_XXXXXY";
  int fd = LIBC_NAMESPACE::mkstemp(tmpl);
  EXPECT_EQ(fd, -1);
  ASSERT_ERRNO_EQ(EINVAL);
}

TEST_F(LlvmLibcMkstempTest, XsNotAtEnd) {
  char tmpl[] = "XXXXXXtmp";
  int fd = LIBC_NAMESPACE::mkstemp(tmpl);
  EXPECT_EQ(fd, -1);
  ASSERT_ERRNO_EQ(EINVAL);
}

TEST_F(LlvmLibcMkstempTest, FiveXsAtEnd) {
  char tmpl[] = "tmp_XXXXX";
  int fd = LIBC_NAMESPACE::mkstemp(tmpl);
  EXPECT_EQ(fd, -1);
  ASSERT_ERRNO_EQ(EINVAL);
}

TEST_F(LlvmLibcMkstempTest, EmptyString) {
  char tmpl[] = "";
  int fd = LIBC_NAMESPACE::mkstemp(tmpl);
  EXPECT_EQ(fd, -1);
  ASSERT_ERRNO_EQ(EINVAL);
}
