// RUN: %clang_csan %s -o %t && %run --threads 64 --blocks 64 %t 2>&1 | FileCheck %s

#include <gpuintrin.h>

#include "race.h"

volatile int data[64];

// CHECK: WARNING: ConcurrencySanitizer: data race
// CHECK: Write of size 4 at 0x{{[0-9a-f]+}}
// CHECK: #0 {{.*array_race\.c:[0-9]+:[0-9]+}}
// CHECK: Address 0x{{[0-9a-f]+}} is global variable 'data'
int main(void) {
  unsigned id = __gpu_num_threads(0) * __gpu_block_id(0) + __gpu_thread_id(0);
  unsigned slot = id % 64;
  RACE_UNTIL_FOUND(i) { data[slot]++; }
  return 0;
}
