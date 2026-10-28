//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tests for mkdtemp
/// See: https://pubs.opengroup.org/onlinepubs/9799919799/functions/mkdtemp.html
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/sys_stat_macros.h"
#include "hdr/types/struct_stat.h"
#include "hdr/unistd_macros.h"
#include "src/__support/CPP/scope.h"
#include "src/__support/CPP/string_view.h"
#include "src/stdlib/mkdtemp.h"
#include "src/string/strdup.h"
#include "src/string/strlen.h"
#include "src/sys/stat/stat.h"
#include "src/unistd/access.h"
#include "src/unistd/rmdir.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LIBC_NAMESPACE::cpp::string_view;
using LlvmLibcMkdtempTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcMkdtempTest, ValidTemplate) {
  char *tmpl = LIBC_NAMESPACE::strdup(libc_make_test_file_path("tmp_XXXXXX"));
  ASSERT_NE(tmpl, nullptr);
  auto cleanup = LIBC_NAMESPACE::cpp::scope_exit([&] {
    LIBC_NAMESPACE::rmdir(tmpl);
    ::free(tmpl);
  });

  ASSERT_THAT(LIBC_NAMESPACE::mkdtemp(tmpl), Succeeds(tmpl));
  EXPECT_THAT(LIBC_NAMESPACE::access(tmpl, F_OK), Succeeds(0));

  struct stat st;
  ASSERT_THAT(LIBC_NAMESPACE::stat(tmpl, &st), Succeeds(0));
  EXPECT_EQ(st.st_mode & S_IFMT, static_cast<mode_t>(S_IFDIR));
  EXPECT_EQ(st.st_mode & (S_IRWXU | S_IRWXG | S_IRWXO),
            static_cast<mode_t>(S_IRWXU));
}

TEST_F(LlvmLibcMkdtempTest, TemplateModifiedInPlace) {
  char *tmpl = LIBC_NAMESPACE::strdup(libc_make_test_file_path("tmp_XXXXXX"));
  ASSERT_NE(tmpl, nullptr);
  auto cleanup_tmpl = LIBC_NAMESPACE::cpp::scope_exit([&] {
    LIBC_NAMESPACE::rmdir(tmpl);
    ::free(tmpl);
  });

  char *orig = LIBC_NAMESPACE::strdup(tmpl);
  ASSERT_NE(orig, nullptr);
  auto cleanup_orig = LIBC_NAMESPACE::cpp::scope_exit([&] { ::free(orig); });

  size_t len = LIBC_NAMESPACE::strlen(tmpl);
  ASSERT_THAT(LIBC_NAMESPACE::mkdtemp(tmpl), Succeeds(tmpl));

  EXPECT_EQ(string_view(tmpl, len - 6), string_view(orig, len - 6));
  EXPECT_NE(string_view(tmpl + len - 6, 6), string_view("XXXXXX"));
}

TEST_F(LlvmLibcMkdtempTest, AllCharactersInCharset) {
  char *tmpl = LIBC_NAMESPACE::strdup(libc_make_test_file_path("tmp_XXXXXX"));
  ASSERT_NE(tmpl, nullptr);
  auto cleanup = LIBC_NAMESPACE::cpp::scope_exit([&] {
    LIBC_NAMESPACE::rmdir(tmpl);
    ::free(tmpl);
  });

  size_t len = LIBC_NAMESPACE::strlen(tmpl);
  ASSERT_THAT(LIBC_NAMESPACE::mkdtemp(tmpl), Succeeds(tmpl));

  // POSIX portable filename character set, sorted by ASCII value.
  // See
  // https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/V1_chap03.html#tag_03_265
  constexpr string_view CHARSET = "-._0123456789"
                                  "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                  "abcdefghijklmnopqrstuvwxyz";
  for (char c : string_view(tmpl + len - 6, 6))
    EXPECT_NE(CHARSET.find_first_of(c), string_view::npos);
}

TEST_F(LlvmLibcMkdtempTest, Uniqueness) {
  char *tmpl1 = LIBC_NAMESPACE::strdup(libc_make_test_file_path("tmp_XXXXXX"));
  ASSERT_NE(tmpl1, nullptr);
  auto cleanup1 = LIBC_NAMESPACE::cpp::scope_exit([&] {
    LIBC_NAMESPACE::rmdir(tmpl1);
    ::free(tmpl1);
  });

  char *tmpl2 = LIBC_NAMESPACE::strdup(libc_make_test_file_path("tmp_XXXXXX"));
  ASSERT_NE(tmpl2, nullptr);
  auto cleanup2 = LIBC_NAMESPACE::cpp::scope_exit([&] {
    LIBC_NAMESPACE::rmdir(tmpl2);
    ::free(tmpl2);
  });

  ASSERT_THAT(LIBC_NAMESPACE::mkdtemp(tmpl1), Succeeds(tmpl1));
  ASSERT_THAT(LIBC_NAMESPACE::mkdtemp(tmpl2), Succeeds(tmpl2));

  EXPECT_STRNE(tmpl1, tmpl2);
}

TEST_F(LlvmLibcMkdtempTest, SixXsNoPrefix) {
  char *tmpl = LIBC_NAMESPACE::strdup(libc_make_test_file_path("XXXXXX"));
  ASSERT_NE(tmpl, nullptr);
  auto cleanup = LIBC_NAMESPACE::cpp::scope_exit([&] {
    LIBC_NAMESPACE::rmdir(tmpl);
    ::free(tmpl);
  });

  ASSERT_THAT(LIBC_NAMESPACE::mkdtemp(tmpl), Succeeds(tmpl));
  EXPECT_THAT(LIBC_NAMESPACE::access(tmpl, F_OK), Succeeds(0));
}

TEST_F(LlvmLibcMkdtempTest, MoreThanSixXs) {
  char *tmpl =
      LIBC_NAMESPACE::strdup(libc_make_test_file_path("tmp_XXXXXXXXXX"));
  ASSERT_NE(tmpl, nullptr);
  auto cleanup_tmpl = LIBC_NAMESPACE::cpp::scope_exit([&] {
    LIBC_NAMESPACE::rmdir(tmpl);
    ::free(tmpl);
  });

  char *orig = LIBC_NAMESPACE::strdup(tmpl);
  ASSERT_NE(orig, nullptr);
  auto cleanup_orig = LIBC_NAMESPACE::cpp::scope_exit([&] { ::free(orig); });

  size_t len = LIBC_NAMESPACE::strlen(tmpl);
  ASSERT_THAT(LIBC_NAMESPACE::mkdtemp(tmpl), Succeeds(tmpl));

  EXPECT_EQ(string_view(tmpl, len - 10), string_view(orig, len - 10));
  EXPECT_NE(string_view(tmpl + len - 10, 10), string_view("XXXXXXXXXX"));
  EXPECT_THAT(LIBC_NAMESPACE::access(tmpl, F_OK), Succeeds(0));
}

#if defined(LIBC_ADD_NULL_CHECKS)
TEST_F(LlvmLibcMkdtempTest, NullPointer) {
  ASSERT_DEATH([] { LIBC_NAMESPACE::mkdtemp(nullptr); }, WITH_SIGNAL(-1));
}
#endif

TEST_F(LlvmLibcMkdtempTest, TemplateTooShort) {
  char tmpl[] = "XXXXX";
  EXPECT_THAT(LIBC_NAMESPACE::mkdtemp(tmpl), Fails<char *>(EINVAL, nullptr));
}

TEST_F(LlvmLibcMkdtempTest, DoesNotEndInXs) {
  char tmpl[] = "tmp_XXXXXY";
  EXPECT_THAT(LIBC_NAMESPACE::mkdtemp(tmpl), Fails<char *>(EINVAL, nullptr));
}

TEST_F(LlvmLibcMkdtempTest, XsNotAtEnd) {
  char tmpl[] = "XXXXXXtmp";
  EXPECT_THAT(LIBC_NAMESPACE::mkdtemp(tmpl), Fails<char *>(EINVAL, nullptr));
}

TEST_F(LlvmLibcMkdtempTest, FiveXsAtEnd) {
  char tmpl[] = "tmp_XXXXX";
  EXPECT_THAT(LIBC_NAMESPACE::mkdtemp(tmpl), Fails<char *>(EINVAL, nullptr));
}

TEST_F(LlvmLibcMkdtempTest, EmptyString) {
  char tmpl[] = "";
  EXPECT_THAT(LIBC_NAMESPACE::mkdtemp(tmpl), Fails<char *>(EINVAL, nullptr));
}

TEST_F(LlvmLibcMkdtempTest, NonExistentParentDirectory) {
  char *tmpl = LIBC_NAMESPACE::strdup(
      libc_make_test_file_path("non_existent_dir/tmp_XXXXXX"));
  ASSERT_NE(tmpl, nullptr);
  auto cleanup = LIBC_NAMESPACE::cpp::scope_exit([&] { ::free(tmpl); });
  EXPECT_THAT(LIBC_NAMESPACE::mkdtemp(tmpl), Fails<char *>(ENOENT, nullptr));
}
