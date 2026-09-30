//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for lgetxattr, lsetxattr, and, llistxattr.
///
//===----------------------------------------------------------------------===//

#include "hdr/sys_stat_macros.h"
#include "hdr/sys_xattr_macros.h"
#include "hdr/types/size_t.h"
#include "hdr/types/ssize_t.h"
#include "src/__support/CPP/scope.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/libc_errno.h"
#include "src/fcntl/creat.h"
#include "src/sys/xattr/lgetxattr.h"
#include "src/sys/xattr/llistxattr.h"
#include "src/sys/xattr/lsetxattr.h"
#include "src/unistd/close.h"
#include "src/unistd/symlink.h"
#include "src/unistd/unlink.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

namespace {

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcLxattrTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;
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

TEST_F(LlvmLibcLxattrTest, GetAndListWithNoAttributeSet) {
  const LIBC_NAMESPACE::CString TEST_FILE_NAME =
      libc_make_test_file_path("testdata/lxattr_unset_testfile.txt");

  int fd = recreate_test_file(TEST_FILE_NAME);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
  scope_exit cleanup([&] {
    ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE_NAME), Succeeds(0));
  });

  constexpr size_t BUFFER_SIZE = 32;
  char buffer[BUFFER_SIZE] = {};

  EXPECT_THAT(LIBC_NAMESPACE::lgetxattr(TEST_FILE_NAME,
                                        "user.missing_test_attr", buffer,
                                        BUFFER_SIZE),
              Fails<ssize_t>(ENODATA));

  EXPECT_THAT(LIBC_NAMESPACE::llistxattr(TEST_FILE_NAME, nullptr, 0),
              Succeeds<ssize_t>(0));

  EXPECT_THAT(LIBC_NAMESPACE::llistxattr(TEST_FILE_NAME, buffer, BUFFER_SIZE),
              Succeeds<ssize_t>(0));
}

TEST_F(LlvmLibcLxattrTest, SetGetAndListWithUserExtendedAttribute) {
  const LIBC_NAMESPACE::CString TEST_FILE_NAME =
      libc_make_test_file_path("testdata/lxattr_testfile.txt");
  constexpr const char *TEST_SYMLINK_TARGET = "lxattr_testfile.txt";
  const LIBC_NAMESPACE::CString TEST_SYMLINK_NAME =
      libc_make_test_file_path("testdata/lxattr_testsymlink.txt");

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

  string_view XATTR_NAME = "user.test_attr";
  string_view XATTR_VALUE = "test_value";
  ASSERT_THAT(LIBC_NAMESPACE::lsetxattr(TEST_FILE_NAME, XATTR_NAME.data(),
                                        XATTR_VALUE.data(), XATTR_VALUE.size(),
                                        /* flags = */ 0),
              Succeeds(0));
  size_t xattr_name_null_terminated_len = XATTR_NAME.size() + 1;

  // Get and list using the test file path.
  EXPECT_THAT(
      LIBC_NAMESPACE::lgetxattr(TEST_FILE_NAME, XATTR_NAME.data(), nullptr, 0),
      Succeeds<ssize_t>(XATTR_VALUE.size()));
  EXPECT_THAT(LIBC_NAMESPACE::llistxattr(TEST_FILE_NAME, nullptr, 0),
              Succeeds<ssize_t>(xattr_name_null_terminated_len));
  {
    constexpr size_t BUFFER_SIZE = 32;
    ASSERT_GE(BUFFER_SIZE, XATTR_NAME.size());

    char buffer[BUFFER_SIZE] = {};
    ASSERT_THAT(LIBC_NAMESPACE::lgetxattr(TEST_FILE_NAME, XATTR_NAME.data(),
                                          buffer, BUFFER_SIZE),
                Succeeds(XATTR_VALUE.size()));
    string_view result_str(buffer, XATTR_VALUE.size());
    EXPECT_EQ(result_str, XATTR_VALUE);
  }
  {
    constexpr size_t BUFFER_SIZE = 32;
    ASSERT_GE(BUFFER_SIZE, xattr_name_null_terminated_len);

    char buffer[BUFFER_SIZE] = {};
    ASSERT_THAT(LIBC_NAMESPACE::llistxattr(TEST_FILE_NAME, buffer, BUFFER_SIZE),
                Succeeds(xattr_name_null_terminated_len));
    string_view result_str(buffer, xattr_name_null_terminated_len - 1);
    EXPECT_EQ(result_str, XATTR_NAME);
  }

  // Get and list using the symlink path to verify the correct syscall is used
  // internally; lgetxattr or llistxattr, and not getxattr or listxattr.
  {
    constexpr size_t BUFFER_SIZE = 32;
    ASSERT_GE(BUFFER_SIZE, xattr_name_null_terminated_len);

    char buffer[BUFFER_SIZE] = {};
    EXPECT_THAT(LIBC_NAMESPACE::lgetxattr(TEST_SYMLINK_NAME, XATTR_NAME.data(),
                                          buffer, BUFFER_SIZE),
                Fails<ssize_t>(ENODATA));
    EXPECT_THAT(
        LIBC_NAMESPACE::llistxattr(TEST_SYMLINK_NAME, buffer, BUFFER_SIZE),
        Succeeds<ssize_t>(0));
  }

  // Get and list with insufficient buffer size, and get with missing attribute
  // on the file to check failing errno.
  {
    constexpr size_t BUFFER_SIZE = 8;
    ASSERT_LT(BUFFER_SIZE, xattr_name_null_terminated_len);
    ASSERT_LT(BUFFER_SIZE, XATTR_VALUE.size());
    char buffer[BUFFER_SIZE] = {};

    EXPECT_THAT(LIBC_NAMESPACE::lgetxattr(TEST_FILE_NAME, XATTR_NAME.data(),
                                          buffer, BUFFER_SIZE),
                Fails<ssize_t>(ERANGE));

    EXPECT_THAT(LIBC_NAMESPACE::llistxattr(TEST_FILE_NAME, buffer, BUFFER_SIZE),
                Fails<ssize_t>(ERANGE));

    EXPECT_THAT(LIBC_NAMESPACE::lgetxattr(TEST_FILE_NAME,
                                          "user.missing_test_attr", buffer,
                                          BUFFER_SIZE),
                Fails<ssize_t>(ENODATA));
  }
}

TEST_F(LlvmLibcLxattrTest, SetAttributeWithNonzeroFlags) {
  const LIBC_NAMESPACE::CString TEST_FILE_NAME =
      libc_make_test_file_path("testdata/lxattr_nonzero_flags_testfile.txt");
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
  ASSERT_THAT(LIBC_NAMESPACE::lsetxattr(TEST_FILE_NAME, XATTR_NAME.data(),
                                        XATTR_VALUE.data(), XATTR_VALUE.size(),
                                        XATTR_REPLACE),
              Fails(ENODATA));

  ASSERT_THAT(LIBC_NAMESPACE::lsetxattr(TEST_FILE_NAME, XATTR_NAME.data(),
                                        XATTR_VALUE.data(), XATTR_VALUE.size(),
                                        /* flags = */ XATTR_CREATE),
              Succeeds(0));

  EXPECT_THAT(LIBC_NAMESPACE::lgetxattr(TEST_FILE_NAME, XATTR_NAME.data(),
                                        buffer, BUFFER_SIZE),
              Succeeds<ssize_t>(XATTR_VALUE.size()));
  ASSERT_EQ(string_view(buffer, XATTR_VALUE.size()), XATTR_VALUE);

  // Replace-only now that the attribute does exist.
  string_view NEW_XATTR_VALUE = "new_test_value";
  ASSERT_THAT(LIBC_NAMESPACE::lsetxattr(TEST_FILE_NAME, XATTR_NAME.data(),
                                        NEW_XATTR_VALUE.data(),
                                        NEW_XATTR_VALUE.size(), XATTR_REPLACE),
              Succeeds(0));

  EXPECT_THAT(LIBC_NAMESPACE::lgetxattr(TEST_FILE_NAME, XATTR_NAME.data(),
                                        buffer, BUFFER_SIZE),
              Succeeds<ssize_t>(NEW_XATTR_VALUE.size()));
  ASSERT_EQ(string_view(buffer, NEW_XATTR_VALUE.size()), NEW_XATTR_VALUE);

  // Try create-only when the attribute already exists.
  ASSERT_THAT(LIBC_NAMESPACE::lsetxattr(TEST_FILE_NAME, XATTR_NAME.data(),
                                        XATTR_VALUE.data(), XATTR_VALUE.size(),
                                        XATTR_CREATE),
              Fails(EEXIST));
}

#if defined(LIBC_ADD_NULL_CHECKS)

TEST_F(LlvmLibcLxattrTest, SetCrashesOnNullPath) {
  EXPECT_DEATH(
      [] {
        constexpr size_t BUFFER_SIZE = 32;
        char buffer[BUFFER_SIZE] = {};
        LIBC_NAMESPACE::lsetxattr(nullptr, "user.attr", buffer, BUFFER_SIZE,
                                  /* flags = */ 0);
      },
      WITH_SIGNAL(-1));
}

TEST_F(LlvmLibcLxattrTest, SetCrashesOnNullAttributeName) {
  EXPECT_DEATH(
      [] {
        constexpr size_t BUFFER_SIZE = 32;
        char buffer[BUFFER_SIZE] = {};
        LIBC_NAMESPACE::lsetxattr("testdata/file.txt", nullptr, buffer,
                                  BUFFER_SIZE, /* flags = */ 0);
      },
      WITH_SIGNAL(-1));
}

TEST_F(LlvmLibcLxattrTest, SetCrashesOnNullBufferNonZeroSize) {
  EXPECT_DEATH(
      [] {
        LIBC_NAMESPACE::lsetxattr("testdata/file.txt", "user.attr", nullptr, 32,
                                  /* flags = */ 0);
      },
      WITH_SIGNAL(-1));
}

TEST_F(LlvmLibcLxattrTest, GetCrashesOnNullPath) {
  EXPECT_DEATH(
      [] {
        constexpr size_t BUFFER_SIZE = 32;
        char buffer[BUFFER_SIZE] = {};
        LIBC_NAMESPACE::lgetxattr(nullptr, "user.attr", buffer, BUFFER_SIZE);
      },
      WITH_SIGNAL(-1));
}

TEST_F(LlvmLibcLxattrTest, GetCrashesOnNullAttributeName) {
  EXPECT_DEATH(
      [] {
        constexpr size_t BUFFER_SIZE = 32;
        char buffer[BUFFER_SIZE] = {};
        LIBC_NAMESPACE::lgetxattr("testdata/file.txt", nullptr, buffer,
                                  BUFFER_SIZE);
      },
      WITH_SIGNAL(-1));
}

TEST_F(LlvmLibcLxattrTest, GetCrashesOnNullBufferNonZeroSize) {
  EXPECT_DEATH(
      [] {
        LIBC_NAMESPACE::lgetxattr("testdata/file.txt", "user.attr", nullptr,
                                  32);
      },
      WITH_SIGNAL(-1));
}

TEST(LlvmLibcLxattrTest, ListCrashesOnNullPath) {
  EXPECT_DEATH(
      [] {
        constexpr size_t BUFFER_SIZE = 32;
        char buffer[BUFFER_SIZE] = {};
        LIBC_NAMESPACE::llistxattr(nullptr, buffer, BUFFER_SIZE);
      },
      WITH_SIGNAL(-1));
}

TEST_F(LlvmLibcLxattrTest, ListCrashesOnNullBufferNonZeroSize) {
  EXPECT_DEATH(
      [] { LIBC_NAMESPACE::llistxattr("testdata/file.txt", nullptr, 32); },
      WITH_SIGNAL(-1));
}

#endif // LIBC_ADD_NULL_CHECKS

} // namespace
