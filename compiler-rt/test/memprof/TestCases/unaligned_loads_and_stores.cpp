// RUN: %clangxx_memprof -O0 %s -o %t && %env_memprof_opts=print_text=true:log_path=stderr %run %t 2>&1 | FileCheck %s

// This is actually:
//  Memory allocation stack id = STACKID
//    alloc_count 1, size (ave/min/max) 128.00 / 128 / 128
// but we need to look for them in the same CHECK to get the correct STACKID.
// CHECK:      Memory allocation stack id = [[STACKID:[0-9]+]]{{[[:space:]].*}}alloc_count 1, size (ave/min/max) 128.00 / 128 / 128
// CHECK-NEXT:   access_count (ave/min/max): 22.00 / 22 / 22

#include <sanitizer/memprof_interface.h>

#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
  // CHECK:      Stack for id [[STACKID]]:
  // CHECK-NEXT:     #0 {{.*}} in operator new[](unsigned long)
  // CHECK-NEXT:     #1 {{.*}} in main {{.*}}:[[@LINE+1]]
  char *x = new char[128];
  memset(x, 0xab, 128);
  __sanitizer_unaligned_load16(x + 15);
  __sanitizer_unaligned_load32(x + 15);
  __sanitizer_unaligned_load64(x + 15);

  __sanitizer_unaligned_store16(x + 15, 0);
  __sanitizer_unaligned_store32(x + 15, 0);
  __sanitizer_unaligned_store64(x + 15, 0);

  delete[] x;
  return 0;
}
