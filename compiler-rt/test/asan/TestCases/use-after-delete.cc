// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%os --check-prefix=CHECK
// RUN: %clangxx_asan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%os --check-prefix=CHECK
// RUN: %clangxx_asan -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%os --check-prefix=CHECK
// RUN: %clangxx_asan -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%os --check-prefix=CHECK
// REQUIRES: stable-runtime

#include <stdlib.h>
int main() {
  char * volatile x = new char[10];
  delete[] x;
  return x[5];
  // CHECK: {{.*ERROR: AddressSanitizer: heap-use-after-free on address}}
  // CHECK:   {{0x.* at pc 0x.* bp 0x.* sp 0x.*}}
  // CHECK: {{READ of size 1 at 0x.* thread T0}}
  // CHECK: {{    #0 0x.* in main .*use-after-delete.cc:}}[[@LINE-4]]
  // CHECK: {{0x.* is located 5 bytes inside of 10-byte region .0x.*,0x.*}}
  // CHECK: {{freed by thread T0 here:}}

  // CHECK-Linux: {{    #0 0x.* in operator delete\[\]}}
  // CHECK-Linux: {{    #1 0x.* in main .*use-after-delete.cc:}}[[@LINE-10]]

  // CHECK: {{previously allocated by thread T0 here:}}

  // CHECK-Linux: {{    #0 0x.* in operator new\[\]}}
  // CHECK-Linux: {{    #1 0x.* in main .*use-after-delete.cc:}}[[@LINE-16]]

  // CHECK: Shadow byte legend (one shadow byte represents {{[0-9]+}} application bytes):
  // CHECK: Global redzone:
  // CHECK: ASan internal:
}
