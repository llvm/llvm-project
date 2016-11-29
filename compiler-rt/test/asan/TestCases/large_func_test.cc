// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%os --check-prefix=CHECK
// RUN: %clangxx_asan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%os --check-prefix=CHECK
// RUN: %clangxx_asan -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%os --check-prefix=CHECK
// RUN: %clangxx_asan -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%os --check-prefix=CHECK
// REQUIRES: stable-runtime

#include <stdlib.h>
__attribute__((noinline))
static void LargeFunction(int *x, int zero) {
  x[0]++;
  x[1]++;
  x[2]++;
  x[3]++;
  x[4]++;
  x[5]++;
  x[6]++;
  x[7]++;
  x[8]++;
  x[9]++;

  // CHECK: {{.*ERROR: AddressSanitizer: heap-buffer-overflow on address}}
  // CHECK:   {{0x.* at pc 0x.* bp 0x.* sp 0x.*}}
  // CHECK: {{READ of size 4 at 0x.* thread T0}}
  x[zero + 103]++;  // we should report this exact line
  // atos incorrectly extracts the symbol name for the static functions on
  // Darwin.
  // CHECK-Linux:  {{#0 0x.* in LargeFunction.*large_func_test.cc:}}[[@LINE-3]]
  // CHECK-Darwin: {{#0 0x.* in .*LargeFunction.*large_func_test.cc}}:[[@LINE-4]]

  x[10]++;
  x[11]++;
  x[12]++;
  x[13]++;
  x[14]++;
  x[15]++;
  x[16]++;
  x[17]++;
  x[18]++;
  x[19]++;
}

int main(int argc, char **argv) {
  int *x = new int[100];
  LargeFunction(x, argc - 1);
  // CHECK: {{    #1 0x.* in main .*large_func_test.cc:}}[[@LINE-1]]
  // CHECK: {{0x.* is located 12 bytes to the right of 400-byte region}}
  // CHECK: {{allocated by thread T0 here:}}
  // CHECK-Linux: {{    #0 0x.* in operator new.*}}
  // CHECK-Darwin: {{    #0 0x.* in .*_Zna.*}}
  // CHECK: {{    #1 0x.* in main .*large_func_test.cc:}}[[@LINE-7]]
  delete[] x;
}
