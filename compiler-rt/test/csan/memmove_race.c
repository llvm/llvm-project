// RUN: %clang_csan %s -o %t && %run --threads 64 --blocks 64 %t 2>&1 | FileCheck %s

#include <gpuintrin.h>

#include "race.h"

#define N 512

int buf[N];

// CHECK: WARNING: ConcurrencySanitizer: data race
// CHECK: {{Write|Read}} of size {{[0-9]+}} at 0x{{[0-9a-f]+}}
// CHECK: #0 {{.*memmove_race\.c:[0-9]+:[0-9]+}}
// CHECK: Address 0x{{[0-9a-f]+}} is global variable 'buf'
int main(void) {
  unsigned id = __gpu_num_threads(0) * __gpu_block_id(0) + __gpu_thread_id(0);
  RACE_UNTIL_FOUND(i) {
    if (id == 0)
      __builtin_memmove((void *)buf, (const void *)(buf + 1),
                        (N - 1) * sizeof(int));
    else
      buf[N - 1] = id;
  }
  return 0;
}
