// RUN: %clangxx_tsan -O1 %s -o %t && %deflake %run %t 2>&1 | FileCheck %s

// Also check that atomics instrumentation can be configured by either driver or
// legacy flags:

// RUN: %clangxx_tsan -O1 %s -o %t -fno-sanitize-thread-atomics && not %deflake %run %t 2>&1 \
// RUN:   | FileCheck --allow-empty --check-prefix=CHECK-NO-ATOMICS %s
// RUN: %clangxx_tsan -O1 %s -o %t -mllvm -tsan-instrument-atomics=0 && not %deflake %run %t 2>&1 \
// RUN:   | FileCheck --allow-empty --check-prefix=CHECK-NO-ATOMICS %s <%t

#include "test.h"

void *Thread(void *a) {
  __atomic_fetch_add((int*)a, 1, __ATOMIC_SEQ_CST);
  barrier_wait(&barrier);
  return 0;
}

int main() {
  barrier_init(&barrier, 2);
  int *a = new int(0);
  pthread_t t;
  pthread_create(&t, 0, Thread, a);
  barrier_wait(&barrier);
  delete a;
  pthread_join(t, 0);
}

// CHECK: WARNING: ThreadSanitizer: data race

// CHECK-NO-ATOMICS-NOT: WARNING: ThreadSanitizer: data race
