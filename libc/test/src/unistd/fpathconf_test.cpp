//===-- Unittests for fpathconf
//------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "src/unistd/close.h"
#include "src/unistd/fpathconf.h"

#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

TEST(LlvmLibcPipeTest, SmokeTest) {
  constexpr const char *FILENAME = "fpathconf.test";
  auto TEST_FILE = libc_make_test_file_path(FILENAME);
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_WRONLY | O_CREAT, S_IRWXU);
  ASSERT_EQ(LIBC_NAMESPACE::pathconf(fd, _PC_SYNC_IO), -1);
  ASSERT_EQ(LIBC_NAMESPACE::pathconf(fd, _PC_PATH_MAX), _POSIX_PATH_MAX);
}

// TODO: Functionality tests
