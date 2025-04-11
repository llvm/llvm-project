//===-- Test handling of malloc TLS data ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/threads/thread.h"
#include "src/stdlib/_Exit.h"
#include "src/stdlib/atexit.h"
#include "src/stdlib/exit.h"
#include "test/IntegrationTest/test.h"

bool cxa_dtor_called = false;
bool fini_dtor_called = false;
bool atexit_called = false;
volatile thread_local int tls_accessible = 0;

extern "C" {
[[gnu::weak]]
void *__dso_handle = nullptr;
int __cxa_thread_atexit_impl(void (*func)(void *), void *arg, void *dso);
void _malloc_thread_cleanup() {
  // make sure that TLS is still alive
  tls_accessible = 1;
  if (!cxa_dtor_called)
    __builtin_trap();
  if (!fini_dtor_called)
    __builtin_trap();
  if (!atexit_called)
    __builtin_trap();
  LIBC_NAMESPACE::_Exit(0);
}
}

[[gnu::destructor]]
void destructor() {
  fini_dtor_called = true;
}

TEST_MAIN() {
  __cxa_thread_atexit_impl([](void *) { cxa_dtor_called = true; }, nullptr,
                           __dso_handle);
  LIBC_NAMESPACE::atexit([]() { atexit_called = true; });
  LIBC_NAMESPACE::exit(1);
}
