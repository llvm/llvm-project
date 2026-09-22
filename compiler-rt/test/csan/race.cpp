// RUN: %clangxx_csan -O1 -g %s -o %t -pthread && %run %t 2>&1 | FileCheck %s

#include "AMDGPU/race.h"
#include <pthread.h>

int Global;

static void *Thread(void *) {
  RACE_UNTIL_FOUND(i)
  Global++;
  return nullptr;
}

int main() {
  pthread_t t;
  pthread_create(&t, nullptr, Thread, nullptr);
  RACE_UNTIL_FOUND(i)
  Global++;
  pthread_join(t, nullptr);
  return 0;
}

// CHECK: WARNING: ConcurrencySanitizer: data race
// CHECK: of size 4 at {{.*}} by thread {{[0-9]+}}:
// CHECK: #{{[0-9]+}} {{.*}} {{main|Thread}}
// CHECK: Location is global 'Global'
