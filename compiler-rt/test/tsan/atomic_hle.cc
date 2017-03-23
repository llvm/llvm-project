// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include "test.h"
#include <sanitizer/tsan_interface_atomic.h>

#ifndef __ATOMIC_HLE_ACQUIRE
#define __ATOMIC_HLE_ACQUIRE (1 << 16)
#endif
#ifndef __ATOMIC_HLE_RELEASE
#define __ATOMIC_HLE_RELEASE (1 << 17)
#endif

int main() {
  volatile int x = 0;
  //__atomic_fetch_add(&x, 1, __ATOMIC_ACQUIRE | __ATOMIC_HLE_ACQUIRE);
  //__atomic_store_n(&x, 0, __ATOMIC_RELEASE | __ATOMIC_HLE_RELEASE);
  __tsan_atomic32_fetch_add(&x, 1,
      (__tsan_memory_order)(__ATOMIC_ACQUIRE | __ATOMIC_HLE_ACQUIRE));
  __tsan_atomic32_store(&x, 0,
      (__tsan_memory_order)(__ATOMIC_RELEASE | __ATOMIC_HLE_RELEASE));
  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK: DONE

