// RUN: %clangxx_csan -O1 -g %s -o %t -pthread
// RUN: not env CSAN_OPTIONS=skip_watch=0:udelay=1000:halt_on_error=1 %run %t 2>&1 | FileCheck %s

#include "AMDGPU/race.h"
#include <pthread.h>

int Global;

static void *Thread(void *) {
  RACE_UNTIL_FOUND(I)
  Global++;
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
