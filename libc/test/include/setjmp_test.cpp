//===-- Unittests for setjmp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDSList-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-macros/offsetof-macro.h"
#include "include/llvm-libc-types/jmp_buf.h"
#include "test/UnitTest/Test.h"

// If this test fails, then *_OFFSET macro definitions in
// libc/src/setjmp/x86_64/setjmp.S need to be fixed. This is a simple change
// detector.
TEST(LlvmLibcSetjmpTest, JmpBufLayout) {
#ifdef __x86_64__
  ASSERT_EQ(offsetof(__jmp_buf, rbx), 0UL);
  ASSERT_EQ(offsetof(__jmp_buf, rbp), 8UL);
  ASSERT_EQ(offsetof(__jmp_buf, r12), 16UL);
  ASSERT_EQ(offsetof(__jmp_buf, r13), 24UL);
  ASSERT_EQ(offsetof(__jmp_buf, r14), 32UL);
  ASSERT_EQ(offsetof(__jmp_buf, r15), 40UL);
  ASSERT_EQ(offsetof(__jmp_buf, rsp), 48UL);
  ASSERT_EQ(offsetof(__jmp_buf, rip), 56UL);
  ASSERT_EQ(sizeof(__jmp_buf), 64UL);
  ASSERT_EQ(alignof(__jmp_buf), 8UL);
#endif // __x86_64__
}
