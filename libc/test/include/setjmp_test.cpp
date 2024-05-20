//===-- Unittests for setjmp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDSList-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-macros/offsetof-macro.h"
#include "include/llvm-libc-types/jmp_buf.h"
#include "src/setjmp/x86_64/setjmp.h"
#include "test/UnitTest/Test.h"

// If this test fails, then *_OFFSET macro definitions in
// libc/src/setjmp/x86_64/setjmp.S need to be fixed. This is a simple change
// detector.
TEST(LlvmLibcSetjmpTest, JmpBufLayout) {
#ifdef __x86_64__
  ASSERT_EQ(offsetof(__jmp_buf, rbx), RBX_OFFSET);
  ASSERT_EQ(offsetof(__jmp_buf, rbp), RBP_OFFSET);
  ASSERT_EQ(offsetof(__jmp_buf, r12), R12_OFFSET);
  ASSERT_EQ(offsetof(__jmp_buf, r13), R13_OFFSET);
  ASSERT_EQ(offsetof(__jmp_buf, r14), R14_OFFSET);
  ASSERT_EQ(offsetof(__jmp_buf, r15), R15_OFFSET);
  ASSERT_EQ(offsetof(__jmp_buf, rsp), RSP_OFFSET);
  ASSERT_EQ(offsetof(__jmp_buf, rip), RIP_OFFSET);
  ASSERT_EQ(sizeof(__jmp_buf), 64UL);
  ASSERT_EQ(alignof(__jmp_buf), 8UL);
#endif // __x86_64__
}
