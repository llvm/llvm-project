//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Test for pthread_join on the main thread.
///
//===----------------------------------------------------------------------===//

#include "src/pthread/pthread_create.h"
#include "src/pthread/pthread_exit.h"
#include "src/pthread/pthread_join.h"
#include "src/pthread/pthread_self.h"
#include "src/stdlib/exit.h"
#include "test/IntegrationTest/test.h"

#include <pthread.h>

static void *const RETVAL = reinterpret_cast<void *>(0xdead);

static pthread_t main_thread;

static void *worker_func(void *) {
  void *retval = nullptr;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_join(main_thread, &retval), 0);
  ASSERT_EQ(retval, RETVAL);

  LIBC_NAMESPACE::exit(0);
  __builtin_unreachable();
}

TEST_MAIN() {
  main_thread = LIBC_NAMESPACE::pthread_self();

  pthread_t worker_thread;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_create(&worker_thread, nullptr, worker_func,
                                           nullptr),
            0);

  LIBC_NAMESPACE::pthread_exit(RETVAL);
  __builtin_unreachable();
}
