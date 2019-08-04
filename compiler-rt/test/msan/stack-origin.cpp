// RUN: %clangxx_msan -O0 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
// RUN: %clangxx_msan -O1 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
// RUN: %clangxx_msan -O2 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
// RUN: %clangxx_msan -O3 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// RUN: %clangxx_msan -fsanitize-memory-track-origins -O0 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-ORIGINS < %t.out
// RUN: %clangxx_msan -fsanitize-memory-track-origins -O1 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-ORIGINS < %t.out
// RUN: %clangxx_msan -fsanitize-memory-track-origins -O2 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-ORIGINS < %t.out
// RUN: %clangxx_msan -fsanitize-memory-track-origins -O3 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out && FileCheck %s --check-prefix=CHECK-ORIGINS < %t.out

#include <stdlib.h>
int main(int argc, char **argv) {
  int x;
  int *volatile p = &x;
  return *p;
  // CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
  // CHECK: {{#0 0x.* in main .*stack-origin.cpp:}}[[@LINE-2]]

  // CHECK-ORIGINS: Uninitialized value was created by an allocation of 'x' in the stack frame of function 'main'
  // CHECK-ORIGINS: {{#0 0x.* in main .*stack-origin.cpp:}}[[@LINE-8]]

  // CHECK: SUMMARY: MemorySanitizer: use-of-uninitialized-value {{.*stack-origin.cpp:.* main}}
}
