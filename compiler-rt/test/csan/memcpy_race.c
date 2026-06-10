// RUN: %clang_csan %s -o %t && %run --threads 64 --blocks 64 %t 2>&1 | FileCheck %s

#include "race.h"

volatile unsigned len = 32;
int dst[64];
int src[64];

// CHECK: WARNING: ConcurrencySanitizer: data race
// CHECK: Write of size {{[0-9]+}} at 0x{{[0-9a-f]+}}
// CHECK: #0 {{.*memcpy_race\.c:[0-9]+:[0-9]+}}
// CHECK: Address 0x{{[0-9a-f]+}} is global variable 'dst'
int main(void) {
  RACE_UNTIL_FOUND(i) { __builtin_memcpy((void *)dst, (const void *)src, len); }
  return 0;
}
