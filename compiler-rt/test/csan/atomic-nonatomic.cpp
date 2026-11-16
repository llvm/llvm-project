// RUN: %clangxx_csan -O1 -g %s -o %t -pthread
// RUN: env CSAN_OPTIONS=skip_watch=0:udelay=1000 %run %t 2>&1 | FileCheck %s

#include "AMDGPU/race.h"
#include <pthread.h>

int Global;

static void *Thread(void *) {
  RACE_UNTIL_FOUND(I)
  __atomic_fetch_add(&Global, 1, __ATOMIC_RELAXED);
  return nullptr;
}

int main() {
  pthread_t T;
  pthread_create(&T, nullptr, Thread, nullptr);
  RACE_UNTIL_FOUND(I)
  Global++;
  pthread_join(T, nullptr);
  return 0;
}

// CHECK: WARNING: ConcurrencySanitizer: data race
// CHECK: Write of size 4
// CHECK: Previous write of size 4
// CHECK: Thread(void*)
