//===-- Unittests for mkdirat ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/stat/mkdirat.h"
#include "src/unistd/rmdir.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include "hdr/fcntl_macros.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcMkdiratTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcMkdiratTest, CreateAndRemove) {
  constexpr const char *FILENAME = "testdata/mkdirat.testdir";
  auto TEST_DIR = libc_make_test_file_path(FILENAME);
  ASSERT_THAT(LIBC_NAMESPACE::mkdirat(AT_FDCWD, TEST_DIR, S_IRWXU),
              Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::rmdir(TEST_DIR), Succeeds(0));
}

TEST_F(LlvmLibcMkdiratTest, BadPath) {
  ASSERT_THAT(
      LIBC_NAMESPACE::mkdirat(AT_FDCWD, "non-existent-dir/test", S_IRWXU),
      Fails(ENOENT));
}
