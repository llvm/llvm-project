//===-- Unittests for mkdirat ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/stat/mkdirat.h"
#include "src/unistd/rmdir.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <fcntl.h>

TEST(LlvmLibcMkdiratTest, CreateAndRemove) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *TEST_DIR = "testdata/mkdirat.testdir";
  ASSERT_THAT(LIBC_NAMESPACE::mkdirat(AT_FDCWD, TEST_DIR, S_IRWXU),
              Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::rmdir(TEST_DIR), Succeeds(0));
}

TEST(LlvmLibcMkdiratTest, BadPath) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  ASSERT_THAT(
      LIBC_NAMESPACE::mkdirat(AT_FDCWD, "non-existent-dir/test", S_IRWXU),
      Fails(ENOENT));
}
