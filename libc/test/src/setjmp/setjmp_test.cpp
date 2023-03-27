//===-- Unittests for setjmp and longjmp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/setjmp/longjmp.h"
#include "src/setjmp/setjmp_impl.h"
#include "test/UnitTest/Test.h"

constexpr int MAX_LOOP = 123;
int longjmp_called = 0;

void jump_back(jmp_buf buf, int n) {
  longjmp_called++;
  __llvm_libc::longjmp(buf, n); // Will return |n| out of setjmp
}

TEST(LlvmLibcSetJmpTest, SetAndJumpBack) {
  jmp_buf buf;
  longjmp_called = 0;

  // Local variables in setjmp scope should be declared volatile.
  volatile int n = 0;
  // The first time setjmp is called, it should return 0.
  // Subsequent calls will return the value passed to jump_back below.
  if (__llvm_libc::setjmp(buf) <= MAX_LOOP) {
    ++n;
    jump_back(buf, n);
  }
  ASSERT_EQ(longjmp_called, n);
  ASSERT_EQ(n, MAX_LOOP + 1);
}

TEST(LlvmLibcSetJmpTest, SetAndJumpBackValOne) {
  jmp_buf buf;
  longjmp_called = 0;

  int val = __llvm_libc::setjmp(buf);
  if (val == 0)
    jump_back(buf, val);

  ASSERT_EQ(longjmp_called, 1);
  ASSERT_EQ(val, 1);
}
