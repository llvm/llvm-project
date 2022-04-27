// RUN: %clangxx_tsan -O1 %s %link_libcxx_tsan -o %t && %run %t 2>&1 | FileCheck %s

#include "test.h"
#include <thread>

int main() {
  barrier_init(&barrier, 2);
  volatile int x = 0;
  std::thread reader([&]() {
    barrier_wait(&barrier);
    int l = x;
    (void)l;
  });
  int cmp = 1;
  __atomic_compare_exchange_n(&x, &cmp, 1, 1, __ATOMIC_RELAXED,
                              __ATOMIC_RELAXED);
  barrier_wait(&barrier);
  reader.join();
  fprintf(stderr, "DONE\n");
}

// CHECK-NOT: ThreadSanitizer: data race
// CHECK: DONE
