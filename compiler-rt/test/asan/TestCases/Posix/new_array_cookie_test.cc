// REQUIRES: asan-64-bits
// RUN: %clangxx_asan -O3 %s -o %t
// RUN:                                    not %run %t 2>&1  | FileCheck %s
// RUN: %env_asan_opts=poison_array_cookie=1 not %run %t 2>&1  | FileCheck %s
// RUN: %env_asan_opts=poison_array_cookie=0 not %run %t 2>&1  | FileCheck %s --check-prefix=NO_COOKIE

// UNSUPPORTED: ios

#include <stdio.h>
#include <stdlib.h>
struct C {
  int x;
  ~C() {
    fprintf(stderr, "ZZZZZZZZ\n");
    exit(1);
  }
};

int main(int argc, char **argv) {
  C *buffer = new C[argc];
  buffer[-2].x = 10;
// CHECK: AddressSanitizer: heap-buffer-overflow
// CHECK: in main {{.*}}new_array_cookie_test.cc:[[@LINE-2]]
// CHECK: is located 0 bytes inside of 12-byte region
// NO_COOKIE: ZZZZZZZZ
  delete [] buffer;
}
