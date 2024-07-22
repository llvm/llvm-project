//===-- Unittests for gettid ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/gettid.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcGetTidTest, SmokeTest) {
  // gettid always succeeds. So, we just call it as a smoke test.
  ASSERT_GT(LIBC_NAMESPACE::gettid(), 0);
}
