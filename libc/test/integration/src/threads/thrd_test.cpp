//===-- Tests for thrd_t creation and joining -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/threads/thrd_create.h"
#include "src/threads/thrd_join.h"

#include "test/IntegrationTest/test.h"

#include <threads.h>

static constexpr int thread_count = 1000;
static int counter = 0;
static int thread_func(void *) {
  ++counter;
  return 0;
}

void create_and_join() {
  for (counter = 0; counter <= thread_count;) {
    thrd_t thread;
    int old_counter_val = counter;
    ASSERT_EQ(__llvm_libc::thrd_create(&thread, thread_func, nullptr),
              (int)thrd_success);
    int retval = thread_count + 1; // Start with a retval we dont expect.
    ASSERT_EQ(__llvm_libc::thrd_join(thread, &retval), (int)thrd_success);
    ASSERT_EQ(retval, 0);
    ASSERT_EQ(counter, old_counter_val + 1);
  }
}

static int return_arg(void *arg) { return *reinterpret_cast<int *>(arg); }

void spawn_and_join() {
  thrd_t thread_list[thread_count];
  int args[thread_count];

  for (int i = 0; i < thread_count; ++i) {
    args[i] = i;
    ASSERT_EQ(__llvm_libc::thrd_create(thread_list + i, return_arg, args + i),
              (int)thrd_success);
  }

  for (int i = 0; i < thread_count; ++i) {
    int retval = thread_count + 1; // Start with a retval we dont expect.
    ASSERT_EQ(__llvm_libc::thrd_join(thread_list[i], &retval),
              (int)thrd_success);
    ASSERT_EQ(retval, i);
  }
}

TEST_MAIN() {
  create_and_join();
  spawn_and_join();
  return 0;
}
