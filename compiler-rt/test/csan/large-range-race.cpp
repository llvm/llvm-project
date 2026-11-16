// RUN: %clangxx_csan -O1 -g %s -o %t -pthread && %run %t 2>&1 | FileCheck %s

#include "AMDGPU/race.h"
#include <pthread.h>

char Global[32768];

static void *Thread(void *) {
  RACE_UNTIL_FOUND(I)
  __builtin_memset(Global, I, sizeof(Global));
  return nullptr;
}

int main() {
  pthread_t T;
  pthread_create(&T, nullptr, Thread, nullptr);
  RACE_UNTIL_FOUND(I)
  __builtin_memset(Global, I, sizeof(Global));
  pthread_join(T, nullptr);
  return 0;
}

// CHECK: WARNING: ConcurrencySanitizer: data race
// CHECK: write of size 8192 at
// CHECK: Location is global 'Global'
