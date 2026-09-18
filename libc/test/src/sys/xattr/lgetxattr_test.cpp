//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for lgetxattr.
///
//===----------------------------------------------------------------------===//

#include "hdr/sys_stat_macros.h"
#include "src/__support/CPP/scope.h"
#include "src/__support/OSUtil/linux/syscall.h"
#include "src/__support/libc_errno.h"
#include "src/fcntl/creat.h"
#include "src/sys/xattr/lgetxattr.h"
#include "src/unistd/close.h"
#include "src/unistd/symlink.h"
#include "src/unistd/unlink.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"
#include <sys/syscall.h>

namespace {

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcLGetxattrTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;
using LIBC_NAMESPACE::cpp::scope_exit;
using LIBC_NAMESPACE::cpp::string_view;

int recreate_test_file(const char *path) {
  LIBC_NAMESPACE::unlink(path);
  LIBC_NAMESPACE::libc_errno = 0;
  return LIBC_NAMESPACE::creat(path, S_IRWXU);
}

int recreate_test_symlink(const char *target, const char *linkpath) {
  LIBC_NAMESPACE::unlink(linkpath);
  LIBC_NAMESPACE::libc_errno = 0;
  return LIBC_NAMESPACE::symlink(target, linkpath);
}

TEST_F(LlvmLibcLGetxattrTest, WithUserExtendedAttribute) {
  const LIBC_NAMESPACE::CString TEST_FILE_NAME =
      libc_make_test_file_path("testdata/lgetxattr.txt");
  constexpr const char *TEST_SYMLINK_TARGET = "lgetxattr.txt";
  const LIBC_NAMESPACE::CString TEST_SYMLINK_NAME =
      libc_make_test_file_path("testdata/lgetxattr_symlink.txt");

  int fd = recreate_test_file(TEST_FILE_NAME);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));

  ASSERT_THAT(recreate_test_symlink(TEST_SYMLINK_TARGET, TEST_SYMLINK_NAME),
              Succeeds(0));
  scope_exit cleanup([&] {
    ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_SYMLINK_NAME), Succeeds(0));
    ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE_NAME), Succeeds(0));
  });

  string_view XATTR_NAME = "user.test_attr";
  string_view XATTR_VALUE = "test_value";
  ASSERT_EQ(0, LIBC_NAMESPACE::syscall_impl<int>(
                   SYS_setxattr, static_cast<const char *>(TEST_FILE_NAME),
                   XATTR_NAME.data(), XATTR_VALUE.data(), XATTR_VALUE.size(),
                   /* flags = */ 0));

  // Call with the test file path.
  {
    EXPECT_THAT(LIBC_NAMESPACE::lgetxattr(TEST_FILE_NAME, XATTR_NAME.data(),
                                          nullptr, 0),
                Succeeds<ssize_t>(XATTR_VALUE.size()));

    constexpr size_t BUFFER_SIZE = 32;
    ASSERT_GE(BUFFER_SIZE, XATTR_VALUE.size());

    char buffer[BUFFER_SIZE] = {};
    ASSERT_THAT(LIBC_NAMESPACE::lgetxattr(TEST_FILE_NAME, XATTR_NAME.data(),
                                          buffer, BUFFER_SIZE),
                Succeeds(XATTR_VALUE.size()));
    string_view result_str(buffer, XATTR_VALUE.size());
    EXPECT_EQ(result_str, XATTR_VALUE);
  }

  // Call with the symlink path to verify the correct syscall is used
  // internally; lgetxattr instead of getxattr.
  {
    constexpr size_t BUFFER_SIZE = 32;
    ASSERT_GE(BUFFER_SIZE, XATTR_VALUE.size());

    char buffer[BUFFER_SIZE] = {};
    EXPECT_THAT(LIBC_NAMESPACE::lgetxattr(TEST_SYMLINK_NAME, XATTR_NAME.data(),
                                          buffer, BUFFER_SIZE),
                Fails<ssize_t>(ENODATA));
  }

  // Call with insufficient buffer size to check failing errno.
  {
    constexpr size_t BUFFER_SIZE = 8;
    ASSERT_LT(BUFFER_SIZE, XATTR_VALUE.size());

    char buffer[BUFFER_SIZE] = {};

    EXPECT_THAT(LIBC_NAMESPACE::lgetxattr(TEST_FILE_NAME, XATTR_NAME.data(),
                                          buffer, BUFFER_SIZE),
                Fails<ssize_t>(ERANGE));
  }
}

#if defined(LIBC_ADD_NULL_CHECKS)

TEST(LlvmLibcLGetxattrTest, CrashOnNullPath) {
  EXPECT_DEATH(
      [] {
        constexpr size_t BUFFER_SIZE = 32;
        char buffer[BUFFER_SIZE] = {};
        LIBC_NAMESPACE::lgetxattr(nullptr, "user.attr", buffer, BUFFER_SIZE);
      },
      WITH_SIGNAL(-1));
}

TEST(LlvmLibcLGetxattrTest, CrashOnNullAttributeName) {
  EXPECT_DEATH(
      [] {
        constexpr size_t BUFFER_SIZE = 32;
        char buffer[BUFFER_SIZE] = {};
        LIBC_NAMESPACE::lgetxattr("testdata/file.txt", nullptr, buffer,
                                  BUFFER_SIZE);
      },
      WITH_SIGNAL(-1));
}

TEST(LlvmLibcLGetxattrTest, CrashOnNullBufferNonZeroSize) {
  EXPECT_DEATH(
      [] {
        LIBC_NAMESPACE::lgetxattr("testdata/file.txt", "user.attr", nullptr,
                                  32);
      },
      WITH_SIGNAL(-1));
}

#endif // LIBC_ADD_NULL_CHECKS

} // namespace
