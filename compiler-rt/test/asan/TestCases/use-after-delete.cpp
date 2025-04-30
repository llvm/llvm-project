// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%os --check-prefix=CHECK
// RUN: %clangxx_asan -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%os --check-prefix=CHECK
// RUN: %clangxx_asan -O2 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%os --check-prefix=CHECK
// RUN: %clangxx_asan -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%os --check-prefix=CHECK
// REQUIRES: stable-runtime

// AIX currently have some issue while symbolizing operator new/delete.
// FIXME: fix this symbolizer issue on aix.

#include <stdlib.h>
int main() {
  char * volatile x = new char[10];
  delete[] x;
  return x[5];
  // CHECK: {{.*ERROR: AddressSanitizer: heap-use-after-free on address}}
  // CHECK:   {{0x.* at pc 0x.* bp 0x.* sp 0x.*}}
  // CHECK: {{READ of size 1 at 0x.* thread T0}}
  // CHECK: {{    #0 0x.* in \.?main .*use-after-delete.cpp:}}[[@LINE-4]]
  // CHECK: {{0x.* is located 5 bytes inside of 10-byte region .0x.*,0x.*}}

  // CHECK: {{freed by thread T0 here:}}
  // CHECK-Linux:  {{    #0 0x.* in operator delete\[\]}}
  // CHECK-SunOS:  {{    #0 0x.* in operator delete\[\]}}
  // CHECK-Windows:{{    #0 0x.* in operator delete\[\]}}
  // CHECK-FreeBSD:{{    #0 0x.* in operator delete\[\]}}
  // CHECK-Darwin: {{    #0 0x.* in .*_Zda}}
  // CHECK-AIX:    {{    #0 0x.*}}
  // CHECK-NEXT:   {{    #1 0x.* in \.?main .*use-after-delete.cpp:}}[[@LINE-15]]

  // CHECK: {{previously allocated by thread T0 here:}}
  // CHECK-Linux:  {{    #0 0x.* in operator new\[\]}}
  // CHECK-SunOS:  {{    #0 0x.* in operator new\[\]}}
  // CHECK-Windows:{{    #0 0x.* in operator new\[\]}}
  // CHECK-FreeBSD:{{    #0 0x.* in operator new\[\]}}
  // CHECK-Darwin: {{    #0 0x.* in .*_Zna}}
  // CHECK-AIX:    {{    #0 0x.*}}
  // CHECK-NEXT:   {{    #1 0x.* in \.?main .*use-after-delete.cpp:}}[[@LINE-25]]

  // CHECK: Shadow byte legend (one shadow byte represents {{[0-9]+}} application bytes):
  // CHECK: Global redzone:
  // CHECK: ASan internal:
}
