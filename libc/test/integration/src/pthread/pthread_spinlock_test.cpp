//===-- Tests for pthread_spinlock ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "src/pthread/pthread_create.h"
#include "src/pthread/pthread_join.h"
#include "src/pthread/pthread_spin_destroy.h"
#include "src/pthread/pthread_spin_init.h"
#include "src/pthread/pthread_spin_lock.h"
#include "src/pthread/pthread_spin_trylock.h"
#include "src/pthread/pthread_spin_unlock.h"
#include "test/IntegrationTest/test.h"
#include <pthread.h>

namespace {
void smoke_test() {
  pthread_spinlock_t lock;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_init(&lock, PTHREAD_PROCESS_PRIVATE),
            0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_lock(&lock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_unlock(&lock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_destroy(&lock), 0);
}

void trylock_test() {
  pthread_spinlock_t lock;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_init(&lock, PTHREAD_PROCESS_PRIVATE),
            0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_trylock(&lock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_trylock(&lock), EBUSY);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_unlock(&lock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_trylock(&lock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_unlock(&lock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_destroy(&lock), 0);
}

void destroy_held_lock_test() {
  pthread_spinlock_t lock;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_init(&lock, PTHREAD_PROCESS_PRIVATE),
            0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_lock(&lock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_destroy(&lock), EBUSY);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_unlock(&lock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_destroy(&lock), 0);
}

void use_after_destroy_test() {
  pthread_spinlock_t lock;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_init(&lock, PTHREAD_PROCESS_PRIVATE),
            0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_destroy(&lock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_unlock(&lock), EINVAL);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_lock(&lock), EINVAL);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_trylock(&lock), EINVAL);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_destroy(&lock), EINVAL);
}

void unlock_without_holding_test() {
  pthread_spinlock_t lock;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_init(&lock, PTHREAD_PROCESS_PRIVATE),
            0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_unlock(&lock), EPERM);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_destroy(&lock), 0);
}

void deadlock_test() {
  pthread_spinlock_t lock;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_init(&lock, PTHREAD_PROCESS_PRIVATE),
            0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_lock(&lock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_lock(&lock), EDEADLK);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_unlock(&lock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_destroy(&lock), 0);
}

void null_lock_test() {
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_init(nullptr, 0), EINVAL);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_lock(nullptr), EINVAL);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_trylock(nullptr), EINVAL);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_unlock(nullptr), EINVAL);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_destroy(nullptr), EINVAL);
}

void pshared_attribute_test() {
  pthread_spinlock_t lock;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_init(&lock, PTHREAD_PROCESS_SHARED),
            0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_destroy(&lock), 0);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_init(&lock, PTHREAD_PROCESS_PRIVATE),
            0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_destroy(&lock), 0);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_init(&lock, -1), EINVAL);
}

void multi_thread_test() {
  struct shared_data {
    pthread_spinlock_t lock;
    int count = 0;
  } shared;
  pthread_t thread[10];
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_init(&shared.lock, 0), 0);
  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(
        LIBC_NAMESPACE::pthread_create(
            &thread[i], nullptr,
            [](void *arg) -> void * {
              auto *data = static_cast<shared_data *>(arg);
              for (int j = 0; j < 1000; ++j) {
                ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_lock(&data->lock), 0);
                data->count += j;
                ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_unlock(&data->lock), 0);
              }
              return nullptr;
            },
            &shared),
        0);
  }
  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(LIBC_NAMESPACE::pthread_join(thread[i], nullptr), 0);
  }
  ASSERT_EQ(LIBC_NAMESPACE::pthread_spin_destroy(&shared.lock), 0);
  ASSERT_EQ(shared.count, 1000 * 999 * 5);
}

} // namespace

TEST_MAIN() {
  smoke_test();
  trylock_test();
  destroy_held_lock_test();
  use_after_destroy_test();
  unlock_without_holding_test();
  deadlock_test();
  multi_thread_test();
  null_lock_test();
  pshared_attribute_test();
  return 0;
}
