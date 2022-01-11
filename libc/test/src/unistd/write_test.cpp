//===-- Unittests for write -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/write.h"
#include "test/ErrnoSetterMatcher.h"
#include "utils/UnitTest/Test.h"
#include "utils/testutils/FDReader.h"

#include <errno.h>

TEST(LlvmLibcUniStd, WriteBasic) {
  using __llvm_libc::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *HELLO = "hello";
  __llvm_libc::testutils::FDReader reader;
  EXPECT_THAT(__llvm_libc::write(reader.get_write_fd(), HELLO, 5), Succeeds(5));
  EXPECT_TRUE(reader.match_written(HELLO));
}

TEST(LlvmLibcUniStd, WriteFails) {
  using __llvm_libc::testing::ErrnoSetterMatcher::Fails;

  EXPECT_THAT(__llvm_libc::write(-1, "", 1), Fails(EBADF));
  EXPECT_THAT(__llvm_libc::write(1, reinterpret_cast<const void *>(-1), 1),
              Fails(EFAULT));
}
