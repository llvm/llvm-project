//===-- Tests for pthread_t -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/pthread/pthread_create.h"
#include "src/pthread/pthread_join.h"
#include "test/IntegrationTest/test.h"

#include <pthread.h>

static constexpr int thread_count = 1000;
static int counter = 0;
static void *thread_func(void *) {
  ++counter;
  return nullptr;
}

void create_and_join() {
  for (counter = 0; counter <= thread_count;) {
    pthread_t thread;
    int old_counter_val = counter;
    ASSERT_EQ(
        __llvm_libc::pthread_create(&thread, nullptr, thread_func, nullptr), 0);

    // Start with a retval we dont expect.
    void *retval = reinterpret_cast<void *>(thread_count + 1);
    ASSERT_EQ(__llvm_libc::pthread_join(thread, &retval), 0);
    ASSERT_EQ(uintptr_t(retval), uintptr_t(nullptr));
    ASSERT_EQ(counter, old_counter_val + 1);
  }
}

static void *return_arg(void *arg) { return arg; }

void spawn_and_join() {
  pthread_t thread_list[thread_count];
  int args[thread_count];

  for (int i = 0; i < thread_count; ++i) {
    args[i] = i;
    ASSERT_EQ(__llvm_libc::pthread_create(thread_list + i, nullptr, return_arg,
                                          args + i),
              0);
  }

  for (int i = 0; i < thread_count; ++i) {
    // Start with a retval we dont expect.
    void *retval = reinterpret_cast<void *>(thread_count + 1);
    ASSERT_EQ(__llvm_libc::pthread_join(thread_list[i], &retval), 0);
    ASSERT_EQ(*reinterpret_cast<int *>(retval), i);
  }
}

TEST_MAIN() {
  create_and_join();
  spawn_and_join();
  return 0;
}
