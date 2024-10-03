//===-- Unittests for fpathconf -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "hdr/fcntl_macros.h"
#include "hdr/limits_macros.h"
#include "hdr/sys_stat_macros.h"
#include "hdr/unistd_macros.h"
#include "src/fcntl/open.h"
#include "src/unistd/close.h"
#include "src/unistd/fpathconf.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

TEST(LlvmLibcPipeTest, SmokeTest) {
  constexpr const char *FILENAME = "fpathconf.test";
  auto TEST_FILE = libc_make_test_file_path(FILENAME);
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_WRONLY | O_CREAT, S_IRWXU);
  EXPECT_EQ(LIBC_NAMESPACE::fpathconf(fd, _PC_SYNC_IO), -1l);
  EXPECT_EQ(LIBC_NAMESPACE::fpathconf(fd, _PC_PATH_MAX),
            static_cast<long>(_POSIX_PATH_MAX));
  LIBC_NAMESPACE::close(fd);
}

// TODO: Functionality tests
