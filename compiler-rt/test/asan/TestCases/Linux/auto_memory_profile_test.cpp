// Tests heap_profile=1.
// Printing memory profiling only works in the configuration where we can
// detect leaks.
// REQUIRES: leak-detection
//
// RUN: %clangxx_asan %s -o %t
// RUN: %env_asan_opts=heap_profile=1 %run %t 2>&1 | FileCheck %s
#include <sanitizer/common_interface_defs.h>

#include <stdio.h>
#include <string.h>
#include <unistd.h>

char *sink[1000];

int main() {

  for (int i = 0; i < 3; i++) {
    const size_t  kSize = 13000000;
    char *x = new char[kSize];
    memset(x, 0, kSize);
    sink[i] = x;
    sleep(1);
  }
}

// CHECK: HEAP PROFILE at RSS
// CHECK: 13000000 byte(s)
// CHECK: HEAP PROFILE at RSS
// CHECK: 26000000 byte(s)
// CHECK: HEAP PROFILE at RSS
// CHECK: 39000000 byte(s)
