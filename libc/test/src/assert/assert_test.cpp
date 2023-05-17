//===-- Unittests for assert ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#undef NDEBUG
#include "src/assert/assert.h"
#include "test/UnitTest/Test.h"

extern "C" int close(int);

TEST(LlvmLibcAssert, Enabled) {
  // -1 matches against any signal, which is necessary for now until
  // __llvm_libc::abort() unblocks SIGABRT. Close standard error for the
  // child process so we don't print the assertion failure message.
  EXPECT_DEATH(
      [] {
        close(2);
        assert(0);
      },
      WITH_SIGNAL(-1));
}

#define NDEBUG
#include "src/assert/assert.h"

TEST(LlvmLibcAssert, Disabled) {
  EXPECT_EXITS([] { assert(0); }, 0);
}
