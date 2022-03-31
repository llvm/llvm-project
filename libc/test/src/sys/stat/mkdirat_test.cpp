//===-- Unittests for mkdirat ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/stat/mkdirat.h"
#include "src/unistd/rmdir.h"
#include "test/ErrnoSetterMatcher.h"
#include "utils/UnitTest/Test.h"
#include "utils/testutils/FDReader.h"

#include <errno.h>
#include <fcntl.h>

TEST(LlvmLibcMkdiratTest, CreateAndRemove) {
  using __llvm_libc::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *TEST_DIR = "testdata/mkdirat.testdir";
  ASSERT_THAT(__llvm_libc::mkdirat(AT_FDCWD, TEST_DIR, S_IRWXU), Succeeds(0));
  ASSERT_THAT(__llvm_libc::rmdir(TEST_DIR), Succeeds(0));
}

TEST(LlvmLibcMkdiratTest, BadPath) {
  using __llvm_libc::testing::ErrnoSetterMatcher::Fails;
  ASSERT_THAT(__llvm_libc::mkdirat(AT_FDCWD, "non-existent-dir/test", S_IRWXU),
              Fails(ENOENT));
}
