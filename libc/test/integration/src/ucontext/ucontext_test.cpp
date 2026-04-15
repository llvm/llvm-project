//===-- Hermetic integration test for ucontext routines -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This is a hermetic integration test for getcontext and setcontext.
// We use a hermetic test here because the heavier unit test infrastructure
// (like GTest) interferes with context switching, stack frame management,
// and floating-point state restoration, causing spurious failures.

#include "test/IntegrationTest/test.h"

#include "include/llvm-libc-types/ucontext_t.h"

#include "src/ucontext/getcontext.h"
#include "src/ucontext/setcontext.h"

void basic_stub_test() {
  ucontext_t ctx;
  static volatile int jumped = 0;

  int ret = LIBC_NAMESPACE::getcontext(&ctx);
  ASSERT_EQ(ret, 0);

  if (!jumped) {
    jumped = 1;
    LIBC_NAMESPACE::setcontext(&ctx);
    ASSERT_TRUE(false); // Should not happen
  }

  ASSERT_TRUE(true);
}

TEST_MAIN() {
  basic_stub_test();
  return 0;
}
