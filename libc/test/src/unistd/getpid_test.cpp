//===-- Unittests for getpid ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/getpid.h"
#include "utils/UnitTest/Test.h"

TEST(LlvmLibcGetPidTest, SmokeTest) {
  // getpid always succeeds. So, we just call it as a smoke test.
  __llvm_libc::getpid();
}
