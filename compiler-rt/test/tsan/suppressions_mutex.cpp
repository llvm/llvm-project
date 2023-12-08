// RUN: %clang_tsan -O1 %s -o %t && %env_tsan_opts=suppressions='%s.supp' %run %t 2>&1 | FileCheck %s
#include "test.h"

void __attribute__((noinline)) suppress_this(pthread_mutex_t *mu) {
  pthread_mutex_destroy(mu);
}

int main() {
  pthread_mutex_t mu;
  pthread_mutex_init(&mu, 0);
  pthread_mutex_lock(&mu);
  suppress_this(&mu);
  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK-NOT: failed to open suppressions file
// CHECK-NOT: WARNING: ThreadSanitizer:
// CHECK: DONE
