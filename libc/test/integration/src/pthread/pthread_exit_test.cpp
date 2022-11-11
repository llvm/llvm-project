//===-- Tests for pthread_exit --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/pthread/pthread_create.h"
#include "src/pthread/pthread_exit.h"
#include "src/pthread/pthread_join.h"
#include "utils/IntegrationTest/test.h"

#include <pthread.h>

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

void *func(void *) {
  // Touch the thread local variable so that it gets initialized and a callback
  // for its destructor gets registered with __cxa_thread_atexit.
  thread_local_a.set(321);
  __llvm_libc::pthread_exit(nullptr);
  return nullptr;
}

TEST_MAIN() {
  pthread_t th;
  void *retval;

  ASSERT_EQ(__llvm_libc::pthread_create(&th, nullptr, func, nullptr), 0);
  ASSERT_EQ(__llvm_libc::pthread_join(th, &retval), 0);

  ASSERT_TRUE(dtor_called);
  __llvm_libc::pthread_exit(nullptr);
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
