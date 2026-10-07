//===-- Test handling of thread local data --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/IntegrationTest/test.h"
#include "src/stdlib/__cxa_thread_atexit_impl.h"

bool called = false;

extern "C" {
[[gnu::weak]]
void *__dso_handle = nullptr;
}

[[gnu::destructor]]
void destructor() {
  if (!called)
    __builtin_trap();
}

TEST_MAIN() {
	LIBC_NAMESPACE::__cxa_thread_atexit_impl([](void *) { called = true; }, nullptr,
                           __dso_handle);
  return 0;
}
