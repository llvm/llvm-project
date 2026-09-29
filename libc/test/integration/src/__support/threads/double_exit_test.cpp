//===-- Test handling of thread local data --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/threads/thread.h"
#include "src/stdlib/exit.h"
#include "test/IntegrationTest/test.h"

extern "C" {
[[gnu::weak]]
void *__dso_handle = nullptr;
int __cxa_thread_atexit_impl(void (*func)(void *), void *arg, void *dso);
}

int call_num = 0;

[[gnu::destructor]]
void check() {
  // This destructor should be called only once.
  if (call_num != 1)
    __builtin_trap();
}

TEST_MAIN() {
  __cxa_thread_atexit_impl([](void *) { LIBC_NAMESPACE::exit(0); }, nullptr,
                           __dso_handle);
  __cxa_thread_atexit_impl([](void *) { ++call_num; }, nullptr, __dso_handle);
  LIBC_NAMESPACE::exit(1);
}
