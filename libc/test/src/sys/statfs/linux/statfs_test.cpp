//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for statfs.
///
//===----------------------------------------------------------------------===//

#include "hdr/fcntl_macros.h"
#include "hdr/sys_stat_macros.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/sys/stat/mkdirat.h"
#include "src/sys/statfs/statfs.h"
#include "src/unistd/rmdir.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcSysStatfsTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcSysStatfsTest, StatfsBasic) {
  struct statfs buf;
  // The root of the file directory must always exist
  ASSERT_THAT(LIBC_NAMESPACE::statfs("/", &buf), Succeeds());
}

TEST_F(LlvmLibcSysStatfsTest, StatfsInvalidPath) {
  struct statfs buf;

  ASSERT_THAT(LIBC_NAMESPACE::statfs("", &buf), Fails(ENOENT));

  // create the file, assert it exists, then delete it and assert it doesn't
  // exist anymore.
  constexpr const char *FILENAME = "statfs.testdir";
  auto TEST_DIR = libc_make_test_file_path(FILENAME);

  // Always delete the folder so that we start in a consistent state.
  LIBC_NAMESPACE::rmdir(TEST_DIR);
  libc_errno = 0; // Reset errno

  ASSERT_THAT(LIBC_NAMESPACE::mkdirat(AT_FDCWD, TEST_DIR, S_IRWXU),
              Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::statfs(TEST_DIR, &buf), Succeeds());

  ASSERT_THAT(LIBC_NAMESPACE::rmdir(TEST_DIR), Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::statfs(TEST_DIR, &buf), Fails(ENOENT));
}
