//===-- Tests for pthread_equal -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/pthread/pthread_create.h"
#include "src/pthread/pthread_equal.h"
#include "src/pthread/pthread_join.h"
#include "src/pthread/pthread_mutex_destroy.h"
#include "src/pthread/pthread_mutex_init.h"
#include "src/pthread/pthread_mutex_lock.h"
#include "src/pthread/pthread_mutex_unlock.h"
#include "src/pthread/pthread_self.h"

#include "test/IntegrationTest/test.h"

#include <pthread.h>
#include <stdint.h> // uintptr_t

pthread_t child_thread;
pthread_mutex_t mutex;

static void *child_func(void *arg) {
  LIBC_NAMESPACE::pthread_mutex_lock(&mutex);
  int *ret = reinterpret_cast<int *>(arg);
  auto self = LIBC_NAMESPACE::pthread_self();
  *ret = LIBC_NAMESPACE::pthread_equal(child_thread, self);
  LIBC_NAMESPACE::pthread_mutex_unlock(&mutex);
  return nullptr;
}

TEST_MAIN() {
  // We init and lock the mutex so that we guarantee that the child thread is
  // waiting after startup.
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_init(&mutex, nullptr), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_lock(&mutex), 0);

  auto main_thread = LIBC_NAMESPACE::pthread_self();

  // The idea here is that, we start a child thread which will immediately
  // wait on |mutex|. The main thread will update the global |child_thread| var
  // and unlock |mutex|. This will give the child thread a chance to compare
  // the result of pthread_self with the |child_thread|. The result of the
  // comparison is returned in the thread arg.
  int result = 0;
  pthread_t th;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_create(&th, nullptr, child_func, &result),
            0);
  // This new thread should of course not be equal to the main thread.
  ASSERT_EQ(LIBC_NAMESPACE::pthread_equal(th, main_thread), 0);

  // Set the |child_thread| global var and unlock to allow the child to perform
  // the comparison.
  child_thread = th;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_unlock(&mutex), 0);

  void *retval;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_join(th, &retval), 0);
  ASSERT_EQ(uintptr_t(retval), uintptr_t(nullptr));
  // The child thread should see that pthread_self return value is the same as
  // |child_thread|.
  ASSERT_NE(result, 0);

  LIBC_NAMESPACE::pthread_mutex_destroy(&mutex);
  return 0;
}
