//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for copy_file_range.
///
//===----------------------------------------------------------------------===//

#include "hdr/fcntl_macros.h"
#include "hdr/sys_stat_macros.h"
#include "src/__support/CPP/string_view.h"
#include "src/fcntl/open.h"
#include "src/unistd/close.h"
#include "src/unistd/copy_file_range.h"
#include "src/unistd/read.h"
#include "src/unistd/unlink.h"
#include "src/unistd/write.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

namespace cpp = LIBC_NAMESPACE::cpp;
using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

class LlvmLibcCopyFileRangeTest
    : public LIBC_NAMESPACE::testing::ErrnoCheckingTest {
protected:
  static constexpr const char *IN_FILENAME = "copy_file_range_in.test";
  static constexpr const char *OUT_FILENAME = "copy_file_range_out.test";
  static constexpr const char TEST_DATA[] = "0123456789ABCDEF";
  static constexpr size_t TEST_SIZE = sizeof(TEST_DATA) - 1;

  decltype(libc_make_test_file_path(IN_FILENAME)) in_path =
      libc_make_test_file_path(IN_FILENAME);
  decltype(libc_make_test_file_path(OUT_FILENAME)) out_path =
      libc_make_test_file_path(OUT_FILENAME);
  int in_fd = -1;
  int out_fd = -1;

  void SetUp() override {
    ErrnoCheckingTest::SetUp();

    int write_fd =
        LIBC_NAMESPACE::open(in_path, O_RDWR | O_CREAT | O_TRUNC, S_IRWXU);
    ASSERT_ERRNO_SUCCESS();
    ASSERT_GT(write_fd, 0);
    ASSERT_EQ(static_cast<ssize_t>(TEST_SIZE),
              LIBC_NAMESPACE::write(write_fd, TEST_DATA, TEST_SIZE));
    ASSERT_THAT(LIBC_NAMESPACE::close(write_fd), Succeeds(0));

    in_fd = LIBC_NAMESPACE::open(in_path, O_RDONLY);
    ASSERT_ERRNO_SUCCESS();
    ASSERT_GT(in_fd, 0);

    out_fd =
        LIBC_NAMESPACE::open(out_path, O_RDWR | O_CREAT | O_TRUNC, S_IRWXU);
    ASSERT_ERRNO_SUCCESS();
    ASSERT_GT(out_fd, 0);
  }

  void TearDown() override {
    if (in_fd >= 0) {
      LIBC_NAMESPACE::close(in_fd);
      in_fd = -1;
    }
    if (out_fd >= 0) {
      LIBC_NAMESPACE::close(out_fd);
      out_fd = -1;
    }
    LIBC_NAMESPACE::unlink(in_path);
    LIBC_NAMESPACE::unlink(out_path);
    ErrnoCheckingTest::TearDown();
  }
};

TEST_F(LlvmLibcCopyFileRangeTest, InvalidArgs) {
  // Bad file descriptors. Under QEMU user mode emulation or restricted
  // environments, nullptr pointers or missing syscalls may report EFAULT
  // or ENOSYS.
  EXPECT_THAT(LIBC_NAMESPACE::copy_file_range(-1, nullptr, -1, nullptr, 10, 0),
              Fails<ssize_t>(any_of(EBADF, EFAULT, ENOSYS)));
}

TEST_F(LlvmLibcCopyFileRangeTest, CopyNoOffsets) {
  EXPECT_THAT(
      LIBC_NAMESPACE::copy_file_range(in_fd, nullptr, out_fd, nullptr, 1, ~0U),
      Fails<ssize_t>(any_of(EINVAL, ENOSYS, EXDEV, EOPNOTSUPP)));

  ssize_t copied = LIBC_NAMESPACE::copy_file_range(in_fd, nullptr, out_fd,
                                                   nullptr, TEST_SIZE, 0);
  if (copied == -1 && (libc_errno == ENOSYS || libc_errno == EXDEV ||
                       libc_errno == EOPNOTSUPP)) {
    // Underlying kernel, container environment (e.g. overlayfs), or emulator
    // does not support copy_file_range.
    return;
  }
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(copied, static_cast<ssize_t>(TEST_SIZE));

  ASSERT_THAT(LIBC_NAMESPACE::close(out_fd), Succeeds(0));
  out_fd = LIBC_NAMESPACE::open(out_path, O_RDONLY);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(out_fd, 0);

  char buf[TEST_SIZE + 1];
  ASSERT_EQ(static_cast<ssize_t>(TEST_SIZE),
            LIBC_NAMESPACE::read(out_fd, buf, TEST_SIZE));
  buf[TEST_SIZE] = '\0';
  EXPECT_EQ(cpp::string_view(buf, TEST_SIZE),
            cpp::string_view(TEST_DATA, TEST_SIZE));
}

TEST_F(LlvmLibcCopyFileRangeTest, CopyWithOffsets) {
  off_t off_in = 4;
  off_t off_out = 0;
  constexpr size_t COPY_LEN = 6;
  ssize_t copied = LIBC_NAMESPACE::copy_file_range(in_fd, &off_in, out_fd,
                                                   &off_out, COPY_LEN, 0);
  if (copied == -1 && (libc_errno == ENOSYS || libc_errno == EXDEV ||
                       libc_errno == EOPNOTSUPP)) {
    return;
  }
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(copied, static_cast<ssize_t>(COPY_LEN));
  EXPECT_EQ(off_in, static_cast<off_t>(10));
  EXPECT_EQ(off_out, static_cast<off_t>(6));

  ASSERT_THAT(LIBC_NAMESPACE::close(out_fd), Succeeds(0));
  out_fd = LIBC_NAMESPACE::open(out_path, O_RDONLY);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GT(out_fd, 0);

  char buf[COPY_LEN + 1];
  ASSERT_EQ(static_cast<ssize_t>(COPY_LEN),
            LIBC_NAMESPACE::read(out_fd, buf, COPY_LEN));
  buf[COPY_LEN] = '\0';
  EXPECT_EQ(cpp::string_view(buf, COPY_LEN), cpp::string_view("456789", 6));
}
