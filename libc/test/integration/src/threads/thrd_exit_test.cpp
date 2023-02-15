//===-- Tests for thrd_exit -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/threads/thrd_create.h"
#include "src/threads/thrd_exit.h"
#include "src/threads/thrd_join.h"
#include "test/IntegrationTest/test.h"

#include <threads.h>

bool dtor_called = false;

class A {
  int val;

public:
  A(int i) { val = i; }

  void set(int i) { val = i; }

  ~A() {
    val = 0;
    dtor_called = true;
  }
};

thread_local A thread_local_a(123);

int func(void *) {
  thread_local_a.set(321);
  __llvm_libc::thrd_exit(0);
  return 0;
}

TEST_MAIN() {
  thrd_t th;
  int retval;

  ASSERT_EQ(__llvm_libc::thrd_create(&th, func, nullptr), thrd_success);
  ASSERT_EQ(__llvm_libc::thrd_join(th, &retval), thrd_success);

  ASSERT_TRUE(dtor_called);
  __llvm_libc::thrd_exit(0);
  return 0;
}

extern "C" {

using Destructor = void(void *);

int __cxa_thread_atexit_impl(Destructor *, void *, void *);

// We do not link integration tests to C++ runtime pieces like the libcxxabi.
// So, we provide our own simple __cxa_thread_atexit implementation.
int __cxa_thread_atexit(Destructor *dtor, void *obj, void *) {
  return __cxa_thread_atexit_impl(dtor, obj, nullptr);
}
}
