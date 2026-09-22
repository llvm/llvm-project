// RUN: %clangxx_csan -O1 -g %s -o %t -pthread
// RUN: env CSAN_OPTIONS=skip_watch=0:udelay=1000 %run %t 1 2>&1 | FileCheck %s --check-prefix=SIZE1
// RUN: env CSAN_OPTIONS=skip_watch=0:udelay=1000 %run %t 2 2>&1 | FileCheck %s --check-prefix=SIZE2
// RUN: env CSAN_OPTIONS=skip_watch=0:udelay=1000 %run %t 8 2>&1 | FileCheck %s --check-prefix=SIZE8
// RUN: env CSAN_OPTIONS=skip_watch=0:udelay=1000 %run %t 16 2>&1 | FileCheck %s --check-prefix=SIZE16

#include "AMDGPU/race.h"
#include <pthread.h>
#include <stdlib.h>

using Vec16 = int __attribute__((vector_size(16)));

#define TEST_SIZE(Name, Type)                                                  \
  volatile Type Name;                                                          \
  static void *Name##Thread(void *) {                                          \
    Type Value = {};                                                           \
    RACE_UNTIL_FOUND(I) Name = Value;                                          \
    return nullptr;                                                            \
  }                                                                            \
  static void Name##Test() {                                                   \
    pthread_t T;                                                               \
    pthread_create(&T, nullptr, Name##Thread, nullptr);                        \
    Type Value = {};                                                           \
    RACE_UNTIL_FOUND(I) Name = Value;                                          \
    pthread_join(T, nullptr);                                                  \
  }

TEST_SIZE(Global1, char)
TEST_SIZE(Global2, short)
TEST_SIZE(Global8, long)
TEST_SIZE(Global16, Vec16)

int main(int Argc, char **Argv) {
  if (Argc != 2)
    return 1;
  switch (atoi(Argv[1])) {
  case 1:
    Global1Test();
    break;
  case 2:
    Global2Test();
    break;
  case 8:
    Global8Test();
    break;
  case 16:
    Global16Test();
    break;
  default:
    return 1;
  }
  return 0;
}

// SIZE1: WARNING: ConcurrencySanitizer: data race
// SIZE1: Write of size 1
// SIZE2: WARNING: ConcurrencySanitizer: data race
// SIZE2: Write of size 2
// SIZE8: WARNING: ConcurrencySanitizer: data race
// SIZE8: Write of size 8
// SIZE16: WARNING: ConcurrencySanitizer: data race
// SIZE16: Write of size 16
