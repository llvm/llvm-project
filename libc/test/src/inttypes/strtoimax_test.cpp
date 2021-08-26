//===-- Unittests for strtoimax -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/inttypes/strtoimax.h"

#include "utils/UnitTest/Test.h"

#include <errno.h>
#include <limits.h>
#include <stddef.h>

// strtoimax is equivalent to strtoll on all currently supported configurations.
// Thus to avoid duplicating code there is just one test to make sure that
// strtoimax works at all. For real tests see stdlib/strtoll_test.cpp.

TEST(LlvmLibcStrToIMaxTest, SimpleCheck) {
  const char *ten = "10";
  errno = 0;
  ASSERT_EQ(__llvm_libc::strtoimax(ten, nullptr, 10), intmax_t(10));
  ASSERT_EQ(errno, 0);
}
