//===-- sanitizer_mutex.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries.
//===----------------------------------------------------------------------===//

#include "sanitizer_mutex.h"

#include "sanitizer_common.h"

namespace __sanitizer {

void Semaphore::Wait() {
  u32 count = atomic_load(&state_, memory_order_relaxed);
  for (;;) {
    if (count == 0) {
      FutexWait(&state_, 0);
      count = atomic_load(&state_, memory_order_relaxed);
      continue;
    }
    if (atomic_compare_exchange_weak(&state_, &count, count - 1,
                                     memory_order_acquire))
      break;
  }
}

void Semaphore::Post(u32 count) {
  CHECK_NE(count, 0);
  atomic_fetch_add(&state_, count, memory_order_release);
  FutexWake(&state_, count);
}

}  // namespace __sanitizer
