//===-- Tests for pthread_rwlock ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/pthread/pthread_rwlock_destroy.h"
#include "src/pthread/pthread_rwlock_init.h"
#include "src/pthread/pthread_rwlock_rdlock.h"
#include "src/pthread/pthread_rwlock_timedrdlock.h"
#include "src/pthread/pthread_rwlock_timedwrlock.h"
#include "src/pthread/pthread_rwlock_tryrdlock.h"
#include "src/pthread/pthread_rwlock_trywrlock.h"
#include "src/pthread/pthread_rwlock_unlock.h"
#include "src/pthread/pthread_rwlock_wrlock.h"

#include "src/pthread/pthread_create.h"
#include "src/pthread/pthread_join.h"

#include "test/IntegrationTest/test.h"

#include <errno.h>
#include <pthread.h>
#include <stdint.h> // uintptr_t

static void smoke_test() {
  pthread_rwlock_t rwlock = PTHREAD_RWLOCK_INITIALIZER;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_init(&rwlock, nullptr), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_rdlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_tryrdlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_trywrlock(&rwlock), EBUSY);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_unlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_unlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_wrlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_rdlock(&rwlock), EDEADLK);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_wrlock(&rwlock), EDEADLK);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_tryrdlock(&rwlock), EBUSY);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_trywrlock(&rwlock), EBUSY);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_unlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_destroy(&rwlock), 0);
}

TEST_MAIN() {
  smoke_test();
  return 0;
}
