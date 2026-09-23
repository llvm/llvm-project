//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for flistxattr.
///
//===----------------------------------------------------------------------===//

#include "hdr/sys_stat_macros.h"
#include "src/__support/CPP/scope.h"
#include "src/__support/OSUtil/linux/syscall.h"
#include "src/__support/libc_errno.h"
#include "src/fcntl/creat.h"
#include "src/sys/xattr/flistxattr.h"
#include "src/unistd/close.h"
#include "src/unistd/unlink.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"
#include <sys/syscall.h>

namespace {

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcListxattrTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;
using LIBC_NAMESPACE::cpp::scope_exit;
using LIBC_NAMESPACE::cpp::string_view;

int recreate_test_file(const char *path) {
  LIBC_NAMESPACE::unlink(path);
  LIBC_NAMESPACE::libc_errno = 0;
  return LIBC_NAMESPACE::creat(path, S_IRWXU);
}

TEST_F(LlvmLibcListxattrTest, NoExtendedAttributes) {
  const LIBC_NAMESPACE::CString TEST_FILE_NAME =
      libc_make_test_file_path("testdata/flistxattr_no_xattrs.txt");
  int fd = recreate_test_file(TEST_FILE_NAME);
  ASSERT_ERRNO_SUCCESS();
  scope_exit cleanup([&] {
    ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
    ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE_NAME), Succeeds(0));
  });

  EXPECT_THAT(LIBC_NAMESPACE::flistxattr(fd, nullptr, 0), Succeeds<ssize_t>(0));

  constexpr size_t BUFFER_SIZE = 32;
  char buffer[BUFFER_SIZE] = {};
  EXPECT_THAT(LIBC_NAMESPACE::flistxattr(fd, buffer, BUFFER_SIZE),
              Succeeds<ssize_t>(0));
}

TEST_F(LlvmLibcListxattrTest, WithUserExtendedAttribute) {
  const LIBC_NAMESPACE::CString TEST_FILE_NAME =
      libc_make_test_file_path("testdata/flistxattr_with_user_xattr.txt");

  int fd = recreate_test_file(TEST_FILE_NAME);
  ASSERT_ERRNO_SUCCESS();
  scope_exit cleanup([&] {
    ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
    ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE_NAME), Succeeds(0));
  });

  string_view XATTR_NAME = "user.test_attr";
  string_view XATTR_VALUE = "test_value";
  ASSERT_EQ(0, LIBC_NAMESPACE::syscall_impl<int>(
                   SYS_fsetxattr, fd, XATTR_NAME.data(), XATTR_VALUE.data(),
                   XATTR_VALUE.size(),
                   /* flags = */ 0));
  size_t xattr_name_null_terminated_len = XATTR_NAME.size() + 1;

  ASSERT_THAT(LIBC_NAMESPACE::flistxattr(fd, nullptr, 0),
              Succeeds<ssize_t>(xattr_name_null_terminated_len));

  {
    constexpr size_t BUFFER_SIZE = 32;
    ASSERT_GE(BUFFER_SIZE, xattr_name_null_terminated_len);

    char buffer[BUFFER_SIZE] = {};
    ASSERT_THAT(LIBC_NAMESPACE::flistxattr(fd, buffer, BUFFER_SIZE),
                Succeeds(xattr_name_null_terminated_len));
    string_view result_str(buffer, xattr_name_null_terminated_len - 1);
    EXPECT_EQ(result_str, XATTR_NAME);
  }

  // Call with insufficient buffer size to check failing errno.
  {
    constexpr size_t BUFFER_SIZE = 14;
    ASSERT_LT(BUFFER_SIZE, xattr_name_null_terminated_len);

    char buffer[BUFFER_SIZE] = {};
    EXPECT_THAT(LIBC_NAMESPACE::flistxattr(fd, buffer, BUFFER_SIZE),
                Fails<ssize_t>(ERANGE));
  }
}

} // namespace
