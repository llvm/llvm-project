//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for setxattr.
///
//===----------------------------------------------------------------------===//

#include "hdr/sys_stat_macros.h"
#include "hdr/sys_xattr_macros.h"
#include "hdr/types/ssize_t.h"
#include "src/__support/CPP/scope.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/OSUtil/linux/syscall.h"
#include "src/__support/libc_errno.h"
#include "src/fcntl/creat.h"
#include "src/sys/xattr/setxattr.h"
#include "src/unistd/close.h"
#include "src/unistd/symlink.h"
#include "src/unistd/unlink.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"
#include <sys/syscall.h>

namespace {

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcSetxattrTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;
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

TEST_F(LlvmLibcSetxattrTest, SetAttributeDefaultFlags) {
  const LIBC_NAMESPACE::CString TEST_FILE_NAME =
      libc_make_test_file_path("testdata/setxattr_default_flags.txt");
  constexpr const char *TEST_SYMLINK_TARGET = "setxattr_default_flags.txt";
  const LIBC_NAMESPACE::CString TEST_SYMLINK_NAME =
      libc_make_test_file_path("testdata/setxattr_default_flags_symlink.txt");

  int fd = recreate_test_file(TEST_FILE_NAME);
  ASSERT_ERRNO_SUCCESS();
  scope_exit unlink_file([&] {
    ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE_NAME), Succeeds(0));
  });
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));

  ASSERT_THAT(recreate_test_symlink(TEST_SYMLINK_TARGET, TEST_SYMLINK_NAME),
              Succeeds(0));
  scope_exit unlink_symlink([&] {
    ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_SYMLINK_NAME), Succeeds(0));
  });

  // Set an attribute through the test file name.
  {
    string_view XATTR_NAME = "user.test_attr_through_file";
    string_view XATTR_VALUE = "test_value_through_file";
    constexpr size_t BUFFER_SIZE = 32;
    ASSERT_GE(BUFFER_SIZE, XATTR_VALUE.size());
    char buffer[BUFFER_SIZE] = {};

    ASSERT_THAT(LIBC_NAMESPACE::setxattr(TEST_FILE_NAME, XATTR_NAME.data(),
                                         XATTR_VALUE.data(), XATTR_VALUE.size(),
                                         /* flags = */ 0),
                Succeeds(0));

    ASSERT_EQ(static_cast<ssize_t>(XATTR_VALUE.size()),
              LIBC_NAMESPACE::syscall_impl<ssize_t>(
                  SYS_getxattr, static_cast<const char *>(TEST_FILE_NAME),
                  XATTR_NAME.data(), buffer, BUFFER_SIZE));
    EXPECT_EQ(string_view(buffer, XATTR_VALUE.size()), XATTR_VALUE);
  }

  // Set another attribute through the symlink to verify the correct syscall is
  // used internally; setxattr instead of lsetxattr.
  {
    string_view XATTR_NAME = "user.test_attr_through_symlink";
    string_view XATTR_VALUE = "test_value_through_symlink";
    constexpr size_t BUFFER_SIZE = 32;
    ASSERT_GE(BUFFER_SIZE, XATTR_VALUE.size());
    char buffer[BUFFER_SIZE] = {};

    ASSERT_THAT(LIBC_NAMESPACE::setxattr(TEST_SYMLINK_NAME, XATTR_NAME.data(),
                                         XATTR_VALUE.data(), XATTR_VALUE.size(),
                                         /* flags = */ 0),
                Succeeds(0));

    EXPECT_EQ(static_cast<ssize_t>(XATTR_VALUE.size()),
              LIBC_NAMESPACE::syscall_impl<ssize_t>(
                  SYS_getxattr, static_cast<const char *>(TEST_FILE_NAME),
                  XATTR_NAME.data(), buffer, BUFFER_SIZE));
    EXPECT_EQ(string_view(buffer, XATTR_VALUE.size()), XATTR_VALUE);
  }
}

TEST_F(LlvmLibcSetxattrTest, SetAttributeWithNonzeroFlags) {
  const LIBC_NAMESPACE::CString TEST_FILE_NAME =
      libc_make_test_file_path("testdata/setxattr_nonzero_flags.txt");
  int fd = recreate_test_file(TEST_FILE_NAME);
  ASSERT_ERRNO_SUCCESS();
  scope_exit cleanup([&] {
    ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE_NAME), Succeeds(0));
  });
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));

  string_view XATTR_NAME = "user.test_attr";
  string_view XATTR_VALUE = "test_value";
  constexpr size_t BUFFER_SIZE = 32;
  ASSERT_GE(BUFFER_SIZE, XATTR_VALUE.size());
  char buffer[BUFFER_SIZE] = {};

  // Try replace-only before attribute exists.
  ASSERT_THAT(LIBC_NAMESPACE::setxattr(TEST_FILE_NAME, XATTR_NAME.data(),
                                       XATTR_VALUE.data(), XATTR_VALUE.size(),
                                       XATTR_REPLACE),
              Fails(ENODATA));

  ASSERT_THAT(LIBC_NAMESPACE::setxattr(TEST_FILE_NAME, XATTR_NAME.data(),
                                       XATTR_VALUE.data(), XATTR_VALUE.size(),
                                       /* flags = */ XATTR_CREATE),
              Succeeds(0));

  ASSERT_EQ(static_cast<ssize_t>(XATTR_VALUE.size()),
            LIBC_NAMESPACE::syscall_impl<ssize_t>(
                SYS_getxattr, static_cast<const char *>(TEST_FILE_NAME),
                XATTR_NAME.data(), buffer, BUFFER_SIZE));
  ASSERT_EQ(string_view(buffer, XATTR_VALUE.size()), XATTR_VALUE);

  // Replace-only now that the attribute exists.
  string_view NEW_XATTR_VALUE = "new_test_value";
  ASSERT_THAT(LIBC_NAMESPACE::setxattr(TEST_FILE_NAME, XATTR_NAME.data(),
                                       NEW_XATTR_VALUE.data(),
                                       NEW_XATTR_VALUE.size(), XATTR_REPLACE),
              Succeeds(0));

  ASSERT_EQ(static_cast<ssize_t>(NEW_XATTR_VALUE.size()),
            LIBC_NAMESPACE::syscall_impl<ssize_t>(
                SYS_getxattr, static_cast<const char *>(TEST_FILE_NAME),
                XATTR_NAME.data(), buffer, BUFFER_SIZE));
  ASSERT_EQ(string_view(buffer, NEW_XATTR_VALUE.size()), NEW_XATTR_VALUE);

  // Try create-only when the attribute already exists.
  ASSERT_THAT(LIBC_NAMESPACE::setxattr(TEST_FILE_NAME, XATTR_NAME.data(),
                                       XATTR_VALUE.data(), XATTR_VALUE.size(),
                                       XATTR_CREATE),
              Fails(EEXIST));
}

#if defined(LIBC_ADD_NULL_CHECKS)

TEST_F(LlvmLibcSetxattrTest, CrashOnNullPath) {
  EXPECT_DEATH(
      [] {
        constexpr size_t BUFFER_SIZE = 32;
        char buffer[BUFFER_SIZE] = {};
        LIBC_NAMESPACE::setxattr(nullptr, "user.attr", buffer, BUFFER_SIZE,
                                 /* flags = */ 0);
      },
      WITH_SIGNAL(-1));
}

TEST_F(LlvmLibcSetxattrTest, CrashOnNullAttributeName) {
  EXPECT_DEATH(
      [] {
        constexpr size_t BUFFER_SIZE = 32;
        char buffer[BUFFER_SIZE] = {};
        LIBC_NAMESPACE::setxattr("testdata/file.txt", nullptr, buffer,
                                 BUFFER_SIZE, /* flags = */ 0);
      },
      WITH_SIGNAL(-1));
}

TEST_F(LlvmLibcSetxattrTest, CrashOnNullBufferNonZeroSize) {
  EXPECT_DEATH(
      [] {
        LIBC_NAMESPACE::setxattr("testdata/file.txt", "user.attr", nullptr, 32,
                                 /* flags = */ 0);
      },
      WITH_SIGNAL(-1));
}

#endif // LIBC_ADD_NULL_CHECKS

} // namespace
