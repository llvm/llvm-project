// RUN: %clangxx_asan -DREAD -O0 -mllvm -asan-dormant %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-READ
// RUN: %clangxx_asan -O0 -mllvm -asan-dormant %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-WRITE
// REQUIRES: stable-runtime

#include <sanitizer/asan_interface.h>
int readHeapAfterFree(){
  int * volatile x = new int[10];
  delete[] x;
  return x[5];
}

static int * volatile y;
int writeHeapAfterFree(){
  y = new int[10];
  delete[] y;
  return y[5] = 413;
}

int main() {
#ifdef READ
  readHeapAfterFree();
  // CHECK-READCHECK-NOT {{.*ERROR: AddressSanitizer: heap-use-after-free on address}}
  __asan_set_dormant(false);
  readHeapAfterFree();
  // CHECK-READ: {{.*ERROR: AddressSanitizer: heap-use-after-free on address}}

#else

  writeHeapAfterFree();
  // CHECK-WRITE-NOT {{.*ERROR: AddressSanitizer: heap-use-after-free on address}}
  __asan_set_dormant(false);
  writeHeapAfterFree();
  // CHECK-WRITE: {{.*ERROR: AddressSanitizer: heap-use-after-free on address}}
#endif

  return 0;
}
