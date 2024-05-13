//===-- Unittests for pthread_rwlockattr_t --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-macros/generic-error-number-macros.h" // EINVAL
#include "src/pthread/pthread_rwlockattr_destroy.h"
#include "src/pthread/pthread_rwlockattr_getpshared.h"
#include "src/pthread/pthread_rwlockattr_init.h"
#include "src/pthread/pthread_rwlockattr_setpshared.h"
#include "test/UnitTest/Test.h"

// TODO: https://github.com/llvm/llvm-project/issues/88997
#include <pthread.h> // PTHREAD_PROCESS_PRIVATE, PTHREAD_PROCESS_SHARED

TEST(LlvmLibcPThreadRWLockAttrTest, InitAndDestroy) {
  pthread_rwlockattr_t attr;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlockattr_init(&attr), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlockattr_destroy(&attr), 0);
}

TEST(LlvmLibcPThreadRWLockAttrTest, GetDefaultValues) {
  pthread_rwlockattr_t attr;

  // Invalid value.
  int pshared = 42;

  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlockattr_init(&attr), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlockattr_getpshared(&attr, &pshared), 0);
  ASSERT_EQ(pshared, PTHREAD_PROCESS_PRIVATE);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlockattr_destroy(&attr), 0);
}

TEST(LlvmLibcPThreadRWLockAttrTest, SetGoodValues) {
  pthread_rwlockattr_t attr;

  // Invalid value.
  int pshared = 42;

  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlockattr_init(&attr), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlockattr_setpshared(
                &attr, PTHREAD_PROCESS_SHARED),
            0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlockattr_getpshared(&attr, &pshared), 0);
  ASSERT_EQ(pshared, PTHREAD_PROCESS_SHARED);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlockattr_destroy(&attr), 0);
}

TEST(LlvmLibcPThreadRWLockAttrTest, SetBadValues) {
  pthread_rwlockattr_t attr;

  // Invalid value.
  int pshared = 42;

  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlockattr_init(&attr), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlockattr_setpshared(&attr, pshared),
            EINVAL);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlockattr_getpshared(&attr, &pshared), 0);
  ASSERT_EQ(pshared, PTHREAD_PROCESS_PRIVATE);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlockattr_destroy(&attr), 0);
}
