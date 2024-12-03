//===-- Unittests for __stack_chk_fail ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/signal_macros.h"
#include "src/__support/macros/sanitizer.h"
#include "src/compiler/__stack_chk_fail.h"
#include "src/string/memset.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStackChkFail, Death) {
  EXPECT_DEATH([] { __stack_chk_fail(); }, WITH_SIGNAL(SIGABRT));
}

// Disable the test when asan is enabled so that it doesn't immediately fail
// after the memset, but before the stack canary is re-checked.
#ifndef LIBC_HAS_ADDRESS_SANITIZER
TEST(LlvmLibcStackChkFail, Smash) {
  EXPECT_DEATH(
      [] {
        int arr[20];
        LIBC_NAMESPACE::memset(arr, 0xAA, 2001);
      },
      WITH_SIGNAL(SIGABRT));
}
#endif // LIBC_HAS_ADDRESS_SANITIZER
