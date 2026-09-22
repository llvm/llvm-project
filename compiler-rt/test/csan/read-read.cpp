// RUN: %clangxx_csan -O1 %s -o %t -pthread
// RUN: env CSAN_OPTIONS=skip_watch=0:udelay=0 %run %t 2>&1 | FileCheck %s --allow-empty

#include <pthread.h>

volatile int Global;

static void *Thread(void *) {
  for (int I = 0; I < 1 << 16; ++I)
    (void)Global;
  return nullptr;
}

int main() {
  pthread_t T;
  pthread_create(&T, nullptr, Thread, nullptr);
  for (int I = 0; I < 1 << 16; ++I)
    (void)Global;
  pthread_join(T, nullptr);
  return 0;
}

// CHECK-NOT: WARNING: ConcurrencySanitizer
