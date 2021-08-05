// RUN: %clang_hwasan %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <stdlib.h>

int main() {
  char *p = (char *)malloc(1);
  free(p + 0x10000000000);
  // CHECK: ERROR: HWAddressSanitizer: invalid-free on address {{.*}} at pc {{[0x]+}}[[PC:.*]] on thread T{{[0-9]+}}
  // CHECK: #0 {{[0x]+}}{{.*}}[[PC]] in free
  // CHECK: #1 {{.*}} in main {{.*}}wild-free.c:[[@LINE-3]]
  // CHECK-NOT: Segmentation fault
  // CHECK-NOT: SIGSEGV
  return 0;
}
