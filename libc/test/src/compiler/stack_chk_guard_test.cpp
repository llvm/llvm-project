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

namespace {

TEST(LlvmLibcStackChkFail, Death) {
  EXPECT_DEATH([] { __stack_chk_fail(); }, WITH_SIGNAL(SIGABRT));
}

// Disable sanitizers such as asan and hwasan that would catch the buffer
// overrun before it clobbered the stack canary word.  Function attributes
// can't be applied to lambdas before C++23, so this has to be separate.  When
// https://github.com/llvm/llvm-project/issues/125760 is fixed, this can use
// the modern spelling [[gnu::no_sanitize(...)]] without conditionalization.
__attribute__((no_sanitize("all"))) void smash_stack() {
  int arr[20];
  LIBC_NAMESPACE::memset(arr, 0xAA, 2001);
}

TEST(LlvmLibcStackChkFail, Smash) {
  EXPECT_DEATH(smash_stack, WITH_SIGNAL(SIGABRT));
}

} // namespace
