// RUN: %clang_csan %s -o %t && %run --threads 64 --blocks 64 %t 2>&1 | FileCheck %s

#include <gpuintrin.h>

#include "race.h"

[[clang::loader_uninitialized]] static volatile __gpu_local int shared[64];

// CHECK: WARNING: ConcurrencySanitizer: data race
// CHECK: {{Write|Read-modify-write|Read}} of size 4 at 0x{{[0-9a-f]+}}
// CHECK: #0 {{.*lds_race\.c:[0-9]+:[0-9]+}}
int main(void) {
  RACE_UNTIL_FOUND(i) { shared[0] += i; }
  return 0;
}
