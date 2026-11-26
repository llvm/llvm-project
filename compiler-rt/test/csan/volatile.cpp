// RUN: %clangxx_csan -O1 -g %s -o %t -pthread
// RUN: env CSAN_OPTIONS=skip_watch=0:udelay=1000 %run %t 2>&1 | FileCheck %s --check-prefix=RACE
// RUN: %clangxx_csan -O1 -g -mllvm -csan-distinguish-volatile %s -o %t -pthread
// RUN: env CSAN_OPTIONS=skip_watch=0:udelay=1000 %run %t 2>&1 | FileCheck %s --check-prefix=MARKED --allow-empty
// RUN: %clangxx_csan -O1 -g -mllvm -csan-distinguish-volatile -DPLAIN %s -o %t -pthread
// RUN: env CSAN_OPTIONS=skip_watch=0:udelay=1000 %run %t 2>&1 | FileCheck %s --check-prefix=RACE
// RUN: %clangxx_csan -O1 -g -mllvm -csan-distinguish-volatile -DUNALIGNED %s -o %t -pthread
// RUN: env CSAN_OPTIONS=skip_watch=0:udelay=1000 %run %t 2>&1 | FileCheck %s --check-prefix=RACE

#include "AMDGPU/race.h"
#include <pthread.h>

struct __attribute__((packed)) Packed {
  char Pad;
  volatile int Value;
};

int Global;
alignas(8) Packed Unaligned;

#if defined(UNALIGNED)
#  define VOLATILE_STORE(V) (Unaligned.Value = (V))
#else
#  define VOLATILE_STORE(V) (*(volatile int *)&Global = (V))
#endif

#if defined(PLAIN)
#  define MAIN_STORE(V) (Global = (V))
#else
#  define MAIN_STORE(V) VOLATILE_STORE(V)
#endif

static void *Thread(void *) {
  RACE_UNTIL_FOUND(I)
  VOLATILE_STORE(I);
  return nullptr;
}

int main() {
  pthread_t T;
  pthread_create(&T, nullptr, Thread, nullptr);
  RACE_UNTIL_FOUND(I)
  MAIN_STORE(I);
  pthread_join(T, nullptr);
  return 0;
}

// RACE: WARNING: ConcurrencySanitizer: data race
// MARKED-NOT: WARNING: ConcurrencySanitizer
