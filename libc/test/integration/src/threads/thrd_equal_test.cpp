//===-- Tests for thrd_equal ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/threads/mtx_destroy.h"
#include "src/threads/mtx_init.h"
#include "src/threads/mtx_lock.h"
#include "src/threads/mtx_unlock.h"
#include "src/threads/thrd_create.h"
#include "src/threads/thrd_current.h"
#include "src/threads/thrd_equal.h"
#include "src/threads/thrd_join.h"

#include "test/IntegrationTest/test.h"

#include <threads.h>

thrd_t child_thread;
mtx_t mutex;

static int child_func(void *arg) {
  LIBC_NAMESPACE::mtx_lock(&mutex);
  int *ret = reinterpret_cast<int *>(arg);
  auto self = LIBC_NAMESPACE::thrd_current();
  *ret = LIBC_NAMESPACE::thrd_equal(child_thread, self);
  LIBC_NAMESPACE::mtx_unlock(&mutex);
  return 0;
}

TEST_MAIN() {
  // We init and lock the mutex so that we guarantee that the child thread is
  // waiting after startup.
  ASSERT_EQ(LIBC_NAMESPACE::mtx_init(&mutex, mtx_plain), int(thrd_success));
  ASSERT_EQ(LIBC_NAMESPACE::mtx_lock(&mutex), int(thrd_success));

  auto main_thread = LIBC_NAMESPACE::thrd_current();

  // The idea here is that, we start a child thread which will immediately
  // wait on |mutex|. The main thread will update the global |child_thread| var
  // and unlock |mutex|. This will give the child thread a chance to compare
  // the result of thrd_self with the |child_thread|. The result of the
  // comparison is returned in the thread arg.
  int result = 0;
  thrd_t th;
  ASSERT_EQ(LIBC_NAMESPACE::thrd_create(&th, child_func, &result),
            int(thrd_success));
  // This new thread should of course not be equal to the main thread.
  ASSERT_EQ(LIBC_NAMESPACE::thrd_equal(th, main_thread), 0);

  // Set the |child_thread| global var and unlock to allow the child to perform
  // the comparison.
  child_thread = th;
  ASSERT_EQ(LIBC_NAMESPACE::mtx_unlock(&mutex), int(thrd_success));

  int retval;
  ASSERT_EQ(LIBC_NAMESPACE::thrd_join(th, &retval), int(thrd_success));
  ASSERT_EQ(retval, 0);
  // The child thread should see that thrd_current return value is the same as
  // |child_thread|.
  ASSERT_NE(result, 0);

  LIBC_NAMESPACE::mtx_destroy(&mutex);
  return 0;
}
