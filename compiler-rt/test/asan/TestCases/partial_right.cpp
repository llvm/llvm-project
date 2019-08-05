// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <stdlib.h>
int main(int argc, char **argv) {
  volatile int *x = (int*)malloc(2*sizeof(int) + 2);
  int res = x[2];  // BOOOM
  // CHECK: {{READ of size 4 at 0x.* thread T0}}
  // CHECK: [[ADDR:0x[01-9a-fa-f]+]] is located 0 bytes to the right of {{.*}}-byte region [{{.*}},{{.*}}[[ADDR]])
  return res;
}
