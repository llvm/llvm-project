//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for fgetxattr, fsetxattr, and, flistxattr.
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
#include "src/sys/xattr/fgetxattr.h"
#include "src/sys/xattr/flistxattr.h"
#include "src/sys/xattr/fsetxattr.h"
#include "src/unistd/close.h"
#include "src/unistd/unlink.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

namespace {

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcFxattrTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;
using LIBC_NAMESPACE::cpp::scope_exit;
using LIBC_NAMESPACE::cpp::string_view;

int recreate_test_file(const char *path) {
  LIBC_NAMESPACE::unlink(path);
  LIBC_NAMESPACE::libc_errno = 0;
  return LIBC_NAMESPACE::creat(path, S_IRWXU);
}

TEST_F(LlvmLibcFxattrTest, GetAndListWithNoAttributeSet) {
  const LIBC_NAMESPACE::CString TEST_FILE_NAME =
      libc_make_test_file_path("testdata/fxattr_no_xattrs.txt");
  int fd = recreate_test_file(TEST_FILE_NAME);
  ASSERT_ERRNO_SUCCESS();
  scope_exit cleanup([&] {
    ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
    ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE_NAME), Succeeds(0));
  });

  constexpr size_t BUFFER_SIZE = 32;
  char buffer[BUFFER_SIZE] = {};

  EXPECT_THAT(LIBC_NAMESPACE::fgetxattr(fd, "user.missing_test_attr", buffer,
                                        BUFFER_SIZE),
              Fails<ssize_t>(ENODATA));

  EXPECT_THAT(LIBC_NAMESPACE::flistxattr(fd, nullptr, 0), Succeeds<ssize_t>(0));
  EXPECT_THAT(LIBC_NAMESPACE::flistxattr(fd, buffer, BUFFER_SIZE),
              Succeeds<ssize_t>(0));
}

TEST_F(LlvmLibcFxattrTest, SetGetAndListWithUserExtendedAttribute) {
  const LIBC_NAMESPACE::CString TEST_FILE_NAME =
      libc_make_test_file_path("testdata/fxattr_testfile.txt");
  int fd = recreate_test_file(TEST_FILE_NAME);
  ASSERT_ERRNO_SUCCESS();
  scope_exit cleanup([&] {
    ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
    ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE_NAME), Succeeds(0));
  });

  string_view XATTR_NAME = "user.test_attr";
  string_view XATTR_VALUE = "test_value";
  ASSERT_THAT(LIBC_NAMESPACE::fsetxattr(fd, XATTR_NAME.data(),
                                        XATTR_VALUE.data(), XATTR_VALUE.size(),
                                        /* flags = */ 0),
              Succeeds(0));
  size_t xattr_name_null_terminated_len = XATTR_NAME.size() + 1;

  // Get and list with null and sufficient buffer size.
  EXPECT_THAT(LIBC_NAMESPACE::fgetxattr(fd, XATTR_NAME.data(), nullptr, 0),
              Succeeds<ssize_t>(XATTR_VALUE.size()));
  EXPECT_THAT(LIBC_NAMESPACE::flistxattr(fd, nullptr, 0),
              Succeeds<ssize_t>(xattr_name_null_terminated_len));
  {
    constexpr size_t BUFFER_SIZE = 32;
    ASSERT_GE(BUFFER_SIZE, XATTR_NAME.size());

    char buffer[BUFFER_SIZE] = {};
    ASSERT_THAT(
        LIBC_NAMESPACE::fgetxattr(fd, XATTR_NAME.data(), buffer, BUFFER_SIZE),
        Succeeds(XATTR_VALUE.size()));
    string_view result_str(buffer, XATTR_VALUE.size());
    EXPECT_EQ(result_str, XATTR_VALUE);
  }
  {
    constexpr size_t BUFFER_SIZE = 32;
    ASSERT_GE(BUFFER_SIZE, xattr_name_null_terminated_len);

    char buffer[BUFFER_SIZE] = {};
    ASSERT_THAT(LIBC_NAMESPACE::flistxattr(fd, buffer, BUFFER_SIZE),
                Succeeds(xattr_name_null_terminated_len));
    string_view result_str(buffer, xattr_name_null_terminated_len - 1);
    EXPECT_EQ(result_str, XATTR_NAME);
  }

  // Get and list with insufficient buffer size, and get with missing attribute
  // to check failing errno.
  {
    constexpr size_t BUFFER_SIZE = 8;
    ASSERT_LT(BUFFER_SIZE, xattr_name_null_terminated_len);
    ASSERT_LT(BUFFER_SIZE, XATTR_VALUE.size());
    char buffer[BUFFER_SIZE] = {};

    EXPECT_THAT(
        LIBC_NAMESPACE::fgetxattr(fd, XATTR_NAME.data(), buffer, BUFFER_SIZE),
        Fails<ssize_t>(ERANGE));

    EXPECT_THAT(LIBC_NAMESPACE::flistxattr(fd, buffer, BUFFER_SIZE),
                Fails<ssize_t>(ERANGE));

    EXPECT_THAT(LIBC_NAMESPACE::fgetxattr(fd, "user.missing_test_attr", buffer,
                                          BUFFER_SIZE),
                Fails<ssize_t>(ENODATA));
  }
}

TEST_F(LlvmLibcFxattrTest, SetAttributeWithNonzeroFlags) {
  const LIBC_NAMESPACE::CString TEST_FILE_NAME =
      libc_make_test_file_path("testdata/fxattr_nonzero_flags_testfile.txt");
  int fd = recreate_test_file(TEST_FILE_NAME);
  ASSERT_ERRNO_SUCCESS();
  scope_exit cleanup([&] {
    ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
    ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE_NAME), Succeeds(0));
  });

  string_view XATTR_NAME = "user.test_attr";
  string_view XATTR_VALUE = "test_value";
  constexpr size_t BUFFER_SIZE = 32;
  ASSERT_GE(BUFFER_SIZE, XATTR_VALUE.size());
  char buffer[BUFFER_SIZE] = {};

  // Try replace-only before attribute exists.
  ASSERT_THAT(LIBC_NAMESPACE::fsetxattr(fd, XATTR_NAME.data(),
                                        XATTR_VALUE.data(), XATTR_VALUE.size(),
                                        XATTR_REPLACE),
              Fails(ENODATA));

  ASSERT_THAT(LIBC_NAMESPACE::fsetxattr(fd, XATTR_NAME.data(),
                                        XATTR_VALUE.data(), XATTR_VALUE.size(),
                                        /* flags = */ XATTR_CREATE),
              Succeeds(0));

  EXPECT_THAT(
      LIBC_NAMESPACE::fgetxattr(fd, XATTR_NAME.data(), buffer, BUFFER_SIZE),
      Succeeds<ssize_t>(XATTR_VALUE.size()));
  ASSERT_EQ(string_view(buffer, XATTR_VALUE.size()), XATTR_VALUE);

  // Replace-only now that the attribute does exist.
  string_view NEW_XATTR_VALUE = "new_test_value";
  ASSERT_THAT(LIBC_NAMESPACE::fsetxattr(fd, XATTR_NAME.data(),
                                        NEW_XATTR_VALUE.data(),
                                        NEW_XATTR_VALUE.size(), XATTR_REPLACE),
              Succeeds(0));

  EXPECT_THAT(
      LIBC_NAMESPACE::fgetxattr(fd, XATTR_NAME.data(), buffer, BUFFER_SIZE),
      Succeeds<ssize_t>(NEW_XATTR_VALUE.size()));
  ASSERT_EQ(string_view(buffer, NEW_XATTR_VALUE.size()), NEW_XATTR_VALUE);

  // Try create-only when the attribute already exists.
  ASSERT_THAT(LIBC_NAMESPACE::fsetxattr(fd, XATTR_NAME.data(),
                                        XATTR_VALUE.data(), XATTR_VALUE.size(),
                                        XATTR_CREATE),
              Fails(EEXIST));
}

#if defined(LIBC_ADD_NULL_CHECKS)

TEST_F(LlvmLibcFxattrTest, SetCrashesOnNullAttributeName) {
  const LIBC_NAMESPACE::CString TEST_FILE_NAME =
      libc_make_test_file_path("testdata/fsetxattr_null_attribute.txt");
  int fd = recreate_test_file(TEST_FILE_NAME);
  ASSERT_ERRNO_SUCCESS();
  scope_exit cleanup([&] {
    ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
    ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE_NAME), Succeeds(0));
  });

  EXPECT_DEATH(
      [fd] {
        constexpr size_t BUFFER_SIZE = 32;
        char buffer[BUFFER_SIZE] = {};
        LIBC_NAMESPACE::fsetxattr(fd, nullptr, buffer, BUFFER_SIZE,
                                  /* flags = */ 0);
      },
      WITH_SIGNAL(-1));
}

TEST_F(LlvmLibcFxattrTest, SetCrashesOnNullBufferNonZeroSize) {
  const LIBC_NAMESPACE::CString TEST_FILE_NAME =
      libc_make_test_file_path("testdata/fsetxattr_null_buffer.txt");
  int fd = recreate_test_file(TEST_FILE_NAME);
  ASSERT_ERRNO_SUCCESS();
  scope_exit cleanup([&] {
    ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
    ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE_NAME), Succeeds(0));
  });

  EXPECT_DEATH(
      [fd] {
        LIBC_NAMESPACE::fsetxattr(fd, "user.attr", nullptr, 32,
                                  /* flags = */ 0);
      },
      WITH_SIGNAL(-1));
}

TEST_F(LlvmLibcFxattrTest, GetCrashesOnNullAttributeName) {
  const LIBC_NAMESPACE::CString TEST_FILE_NAME =
      libc_make_test_file_path("testdata/fgetxattr_null_attribute.txt");
  int fd = recreate_test_file(TEST_FILE_NAME);
  ASSERT_ERRNO_SUCCESS();
  scope_exit cleanup([&] {
    ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
    ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE_NAME), Succeeds(0));
  });

  EXPECT_DEATH(
      [fd] {
        constexpr size_t BUFFER_SIZE = 32;
        char buffer[BUFFER_SIZE] = {};
        LIBC_NAMESPACE::fgetxattr(fd, nullptr, buffer, BUFFER_SIZE);
      },
      WITH_SIGNAL(-1));
}

TEST_F(LlvmLibcFxattrTest, GetCrashesOnNullBufferNonZeroSize) {
  const LIBC_NAMESPACE::CString TEST_FILE_NAME =
      libc_make_test_file_path("testdata/fgetxattr_null_buffer.txt");
  int fd = recreate_test_file(TEST_FILE_NAME);
  ASSERT_ERRNO_SUCCESS();
  scope_exit cleanup([&] {
    ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
    ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE_NAME), Succeeds(0));
  });

  EXPECT_DEATH(
      [fd] { LIBC_NAMESPACE::fgetxattr(fd, "user.attr", nullptr, 32); },
      WITH_SIGNAL(-1));
}

TEST_F(LlvmLibcFxattrTest, ListCrashesOnNullBufferNonZeroSize) {
  const LIBC_NAMESPACE::CString TEST_FILE_NAME =
      libc_make_test_file_path("testdata/fgetxattr_null_buffer.txt");
  int fd = recreate_test_file(TEST_FILE_NAME);
  ASSERT_ERRNO_SUCCESS();
  scope_exit cleanup([&] {
    ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
    ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE_NAME), Succeeds(0));
  });

  EXPECT_DEATH([fd] { LIBC_NAMESPACE::flistxattr(fd, nullptr, 32); },
               WITH_SIGNAL(-1));
}

#endif // LIBC_ADD_NULL_CHECKS

} // namespace
