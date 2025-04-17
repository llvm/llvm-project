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
#include "src/string/memory_utils/inline_memcmp.h"
#include "src/string/memory_utils/inline_memset.h"
#include "test/UnitTest/Test.h"

constexpr int MAX_LOOP = 123;
int longjmp_called = 0;

void jump_back(jmp_buf buf, int n) {
  longjmp_called++;
  LIBC_NAMESPACE::siglongjmp(buf, n); // Will return |n| out of setjmp
}

#define SMOKE_TESTS(SUFFIX, FLAG)                                              \
  TEST(LlvmLibcSetJmpTest, SigSetAndJumpBack##SUFFIX) {                        \
    jmp_buf buf;                                                               \
    longjmp_called = 0;                                                        \
    volatile int n = 0;                                                        \
    sigset_t old;                                                              \
    sigset_t mask_all;                                                         \
    sigset_t recovered;                                                        \
    LIBC_NAMESPACE::inline_memset(&mask_all, 0xFF, sizeof(mask_all));          \
    LIBC_NAMESPACE::inline_memset(&old, 0, sizeof(old));                       \
    LIBC_NAMESPACE::inline_memset(&recovered, 0, sizeof(recovered));           \
    LIBC_NAMESPACE::sigprocmask(0, nullptr, &old);                             \
    if (LIBC_NAMESPACE::sigsetjmp(buf, FLAG) <= MAX_LOOP) {                    \
      if (FLAG) {                                                              \
        LIBC_NAMESPACE::sigprocmask(0, nullptr, &recovered);                   \
        ASSERT_EQ(                                                             \
            0, LIBC_NAMESPACE::inline_memcmp(&old, &recovered, sizeof(old)));  \
      }                                                                        \
      n = n + 1;                                                               \
      if (FLAG)                                                                \
        LIBC_NAMESPACE::sigprocmask(SIG_BLOCK, &mask_all, nullptr);            \
      jump_back(buf, n);                                                       \
    }                                                                          \
    ASSERT_EQ(longjmp_called, n);                                              \
    ASSERT_EQ(n, MAX_LOOP + 1);                                                \
  }                                                                            \
  TEST(LlvmLibcSetJmpTest, SigSetAndJumpBackValOne##SUFFIX) {                  \
    jmp_buf buf;                                                               \
    longjmp_called = 0;                                                        \
    sigset_t old;                                                              \
    sigset_t mask_all;                                                         \
    sigset_t recovered;                                                        \
    LIBC_NAMESPACE::inline_memset(&mask_all, 0xFF, sizeof(mask_all));          \
    LIBC_NAMESPACE::inline_memset(&old, 0, sizeof(old));                       \
    LIBC_NAMESPACE::inline_memset(&recovered, 0, sizeof(recovered));           \
    LIBC_NAMESPACE::sigprocmask(0, nullptr, &old);                             \
    int val = LIBC_NAMESPACE::sigsetjmp(buf, FLAG);                            \
    if (val == 0) {                                                            \
      if (FLAG)                                                                \
        LIBC_NAMESPACE::sigprocmask(SIG_BLOCK, &mask_all, nullptr);            \
      jump_back(buf, val);                                                     \
    }                                                                          \
    if (FLAG) {                                                                \
      LIBC_NAMESPACE::sigprocmask(0, nullptr, &recovered);                     \
      ASSERT_EQ(0,                                                             \
                LIBC_NAMESPACE::inline_memcmp(&old, &recovered, sizeof(old))); \
    }                                                                          \
    ASSERT_EQ(longjmp_called, 1);                                              \
    ASSERT_EQ(val, 1);                                                         \
  }

SMOKE_TESTS(SaveSigs, 1)
SMOKE_TESTS(NoSaveSigs, 0)
