//===-- Tests for __cxa_thread_atexit_impl --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/pthread/pthread_create.h"
#include "src/pthread/pthread_exit.h"
#include "src/pthread/pthread_join.h"
#include "src/stdlib/__cxa_thread_atexit_impl.h"
#include "test/IntegrationTest/test.h"

#include <pthread.h>

namespace {

int dtor_order[3];
int dtor_count = 0;

struct Obj {
  int id;
  Obj(int i) : id(i) {}
  ~Obj() { dtor_order[dtor_count++] = id; }
};

thread_local Obj tl_first(1);
thread_local Obj tl_second(2);
thread_local Obj tl_third(3);

void touch_all() {
  // touch is in definition order so the destruction order becomes well defined.
  (void)tl_first.id;
  (void)tl_second.id;
  (void)tl_third.id;
}

void *return_func(void *) {
  touch_all();
  return nullptr;
}

void *exit_func(void *) {
  touch_all();
  LIBC_NAMESPACE::pthread_exit(nullptr);
  return nullptr;
}

void run_thread(void *(*fn)(void *)) {
  dtor_count = 0;

  pthread_t th;
  void *retval;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_create(&th, nullptr, fn, nullptr), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_join(th, &retval), 0);

  ASSERT_EQ(dtor_count, 3);
  ASSERT_EQ(dtor_order[0], 3);
  ASSERT_EQ(dtor_order[1], 2);
  ASSERT_EQ(dtor_order[2], 1);
}

} // namespace

extern "C" {
// Integration tests are not linked to the C++ runtime, so provide a minimal
// __cxa_thread_atexit for the compiler-emitted calls.
int __cxa_thread_atexit(void (*dtor)(void *), void *obj, void *) {
  return LIBC_NAMESPACE::__cxa_thread_atexit_impl(dtor, obj, nullptr);
}
}

TEST_MAIN() {
  run_thread(return_func);
  run_thread(exit_func);
  return 0;
}
