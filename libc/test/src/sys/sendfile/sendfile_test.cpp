//===-- Unittests for sendfile --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/string_view.h"
#include "src/errno/libc_errno.h"
#include "src/fcntl/open.h"
#include "src/sys/sendfile/sendfile.h"
#include "src/unistd/close.h"
#include "src/unistd/read.h"
#include "src/unistd/unlink.h"
#include "src/unistd/write.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <fcntl.h>
#include <sys/stat.h>

namespace cpp = LIBC_NAMESPACE::cpp;

TEST(LlvmLibcSendfileTest, CreateAndTransfer) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

  // The test strategy is to
  //   1. Create a temporary file with known data.
  //   2. Use sendfile to copy it to another file.
  //   3. Make sure that the data was actually copied.
  //   4. Clean up the temporary files.
  constexpr const char *IN_FILE = "testdata/sendfile_in.test";
  constexpr const char *OUT_FILE = "testdata/sendfile_out.test";
  const char IN_DATA[] = "sendfile test";
  constexpr ssize_t IN_SIZE = ssize_t(sizeof(IN_DATA));
  LIBC_NAMESPACE::libc_errno = 0;

  int in_fd = LIBC_NAMESPACE::open(IN_FILE, O_CREAT | O_WRONLY, S_IRWXU);
  ASSERT_GT(in_fd, 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(LIBC_NAMESPACE::write(in_fd, IN_DATA, IN_SIZE), IN_SIZE);
  ASSERT_THAT(LIBC_NAMESPACE::close(in_fd), Succeeds(0));

  in_fd = LIBC_NAMESPACE::open(IN_FILE, O_RDONLY);
  ASSERT_GT(in_fd, 0);
  ASSERT_ERRNO_SUCCESS();
  int out_fd = LIBC_NAMESPACE::open(OUT_FILE, O_CREAT | O_WRONLY, S_IRWXU);
  ASSERT_GT(out_fd, 0);
  ASSERT_ERRNO_SUCCESS();
  ssize_t size = LIBC_NAMESPACE::sendfile(in_fd, out_fd, nullptr, IN_SIZE);
  ASSERT_EQ(size, IN_SIZE);
  ASSERT_THAT(LIBC_NAMESPACE::close(in_fd), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::close(out_fd), Succeeds(0));

  out_fd = LIBC_NAMESPACE::open(OUT_FILE, O_RDONLY);
  ASSERT_GT(out_fd, 0);
  ASSERT_ERRNO_SUCCESS();
  char buf[IN_SIZE];
  ASSERT_EQ(IN_SIZE, LIBC_NAMESPACE::read(out_fd, buf, IN_SIZE));
  ASSERT_EQ(cpp::string_view(buf), cpp::string_view(IN_DATA));

  ASSERT_THAT(LIBC_NAMESPACE::unlink(IN_FILE), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::unlink(OUT_FILE), Succeeds(0));
}
