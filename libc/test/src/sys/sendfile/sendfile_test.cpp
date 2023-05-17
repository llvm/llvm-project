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
#include "test/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <fcntl.h>
#include <sys/stat.h>

namespace cpp = __llvm_libc::cpp;

TEST(LlvmLibcSendfileTest, CreateAndTransfer) {
  using __llvm_libc::testing::ErrnoSetterMatcher::Fails;
  using __llvm_libc::testing::ErrnoSetterMatcher::Succeeds;

  // The test strategy is to
  //   1. Create a temporary file with known data.
  //   2. Use sendfile to copy it to another file.
  //   3. Make sure that the data was actually copied.
  //   4. Clean up the temporary files.
  constexpr const char *IN_FILE = "testdata/sendfile_in.test";
  constexpr const char *OUT_FILE = "testdata/sendfile_out.test";
  const char IN_DATA[] = "sendfile test";
  constexpr ssize_t IN_SIZE = ssize_t(sizeof(IN_DATA));
  libc_errno = 0;

  int in_fd = __llvm_libc::open(IN_FILE, O_CREAT | O_WRONLY, S_IRWXU);
  ASSERT_GT(in_fd, 0);
  ASSERT_EQ(libc_errno, 0);
  ASSERT_EQ(__llvm_libc::write(in_fd, IN_DATA, IN_SIZE), IN_SIZE);
  ASSERT_THAT(__llvm_libc::close(in_fd), Succeeds(0));

  in_fd = __llvm_libc::open(IN_FILE, O_RDONLY);
  ASSERT_GT(in_fd, 0);
  ASSERT_EQ(libc_errno, 0);
  int out_fd = __llvm_libc::open(OUT_FILE, O_CREAT | O_WRONLY, S_IRWXU);
  ASSERT_GT(out_fd, 0);
  ASSERT_EQ(libc_errno, 0);
  ssize_t size = __llvm_libc::sendfile(in_fd, out_fd, nullptr, IN_SIZE);
  ASSERT_EQ(size, IN_SIZE);
  ASSERT_THAT(__llvm_libc::close(in_fd), Succeeds(0));
  ASSERT_THAT(__llvm_libc::close(out_fd), Succeeds(0));

  out_fd = __llvm_libc::open(OUT_FILE, O_RDONLY);
  ASSERT_GT(out_fd, 0);
  ASSERT_EQ(libc_errno, 0);
  char buf[IN_SIZE];
  ASSERT_EQ(IN_SIZE, __llvm_libc::read(out_fd, buf, IN_SIZE));
  ASSERT_EQ(cpp::string_view(buf), cpp::string_view(IN_DATA));

  ASSERT_THAT(__llvm_libc::unlink(IN_FILE), Succeeds(0));
  ASSERT_THAT(__llvm_libc::unlink(OUT_FILE), Succeeds(0));
}
