// RUN: %clangxx_csan -O1 %s -o %t -pthread && %run %t 2>&1 | FileCheck %s --allow-empty

#include <pthread.h>

int Global;

static void *Thread(void *) {
  for (int I = 0; I < 1024; ++I)
    __atomic_fetch_add(&Global, 1, __ATOMIC_RELAXED);
  return nullptr;
}

int main() {
  pthread_t t;
  pthread_create(&t, nullptr, Thread, nullptr);
  for (int I = 0; I < 1024; ++I)
    __atomic_fetch_add(&Global, 1, __ATOMIC_RELAXED);
  pthread_join(t, nullptr);
  return 0;
}

// CHECK-NOT: WARNING: ConcurrencySanitizer
