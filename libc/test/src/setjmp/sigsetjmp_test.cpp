//===-- Unittests for sigsetjmp and siglongjmp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/setjmp/siglongjmp.h"
#include "src/setjmp/sigsetjmp.h"
#include "src/signal/sigprocmask.h"
#include "src/string/memcmp.h"
#include "src/string/memset.h"
#include "test/UnitTest/Test.h"

constexpr int MAX_LOOP = 123;
int longjmp_called = 0;

void jump_back(jmp_buf buf, int n) {
  longjmp_called++;
  LIBC_NAMESPACE::siglongjmp(buf, n); // Will return |n| out of setjmp
}

TEST(LlvmLibcSetJmpTest, SigSetAndJumpBackSaveSigs) {
  jmp_buf buf;
  longjmp_called = 0;
  volatile int n = 0;
  sigset_t old;
  sigset_t mask_all;
  sigset_t recovered;
  LIBC_NAMESPACE::memset(&mask_all, 0xFF, sizeof(mask_all));
  LIBC_NAMESPACE::memset(&old, 0, sizeof(old));
  LIBC_NAMESPACE::memset(&recovered, 0, sizeof(recovered));
  LIBC_NAMESPACE::sigprocmask(0, nullptr, &old);
  if (LIBC_NAMESPACE::sigsetjmp(buf, 1) <= MAX_LOOP) {
    LIBC_NAMESPACE::sigprocmask(0, nullptr, &recovered);
    ASSERT_EQ(0, LIBC_NAMESPACE::memcmp(&old, &recovered, sizeof(old)));
    n = n + 1;
    LIBC_NAMESPACE::sigprocmask(SIG_BLOCK, &mask_all, nullptr);
    jump_back(buf, n);
  }
  ASSERT_EQ(longjmp_called, n);
  ASSERT_EQ(n, MAX_LOOP + 1);
}

TEST(LlvmLibcSetJmpTest, SigSetAndJumpBackValOneSaveSigs) {
  jmp_buf buf;
  longjmp_called = 0;
  sigset_t old;
  sigset_t mask_all;
  sigset_t recovered;
  LIBC_NAMESPACE::memset(&mask_all, 0xFF, sizeof(mask_all));
  LIBC_NAMESPACE::memset(&old, 0, sizeof(old));
  LIBC_NAMESPACE::memset(&recovered, 0, sizeof(recovered));
  LIBC_NAMESPACE::sigprocmask(0, nullptr, &old);
  int val = LIBC_NAMESPACE::sigsetjmp(buf, 1);
  if (val == 0) {
    LIBC_NAMESPACE::sigprocmask(SIG_BLOCK, &mask_all, nullptr);
    jump_back(buf, val);
  }
  LIBC_NAMESPACE::sigprocmask(0, nullptr, &recovered);
  ASSERT_EQ(0, LIBC_NAMESPACE::memcmp(&old, &recovered, sizeof(old)));
  ASSERT_EQ(longjmp_called, 1);
  ASSERT_EQ(val, 1);
}

TEST(LlvmLibcSetJmpTest, SigSetAndJumpBackNoSaveSigs) {
  jmp_buf buf;
  longjmp_called = 0;
  volatile int n = 0;
  if (LIBC_NAMESPACE::sigsetjmp(buf, 0) <= MAX_LOOP) {
    n = n + 1;
    jump_back(buf, n);
  }
  ASSERT_EQ(longjmp_called, n);
  ASSERT_EQ(n, MAX_LOOP + 1);
}

TEST(LlvmLibcSetJmpTest, SigSetAndJumpBackValOneNoSaveSigs) {
  jmp_buf buf;
  longjmp_called = 0;
  int val = LIBC_NAMESPACE::sigsetjmp(buf, 0);
  if (val == 0) {
    jump_back(buf, val);
  }
  ASSERT_EQ(longjmp_called, 1);
  ASSERT_EQ(val, 1);
}
