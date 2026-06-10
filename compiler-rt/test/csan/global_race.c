// RUN: %clang_csan %s -o %t && %run --threads 64 --blocks 64 %t 2>&1 | FileCheck %s

#include "race.h"

volatile int global;

// CHECK: WARNING: ConcurrencySanitizer: data race
// CHECK: Write of size 4 at 0x{{[0-9a-f]+}}
// CHECK: #0 {{.*global_race\.c:[0-9]+:[0-9]+}}
// CHECK: Address 0x{{[0-9a-f]+}} is global variable 'global'
int main(void) {
  RACE_UNTIL_FOUND(i) { global++; }
  return 0;
}
