// RUN: %clangxx_msan -fno-sanitize-memory-param-retval -fsanitize-memory-track-origins=2 -O3 %s -o %t && \
// RUN:     env MSAN_OPTIONS=store_context_size=1 not %run %t 2>&1 | FileCheck %s

// Test that stack trace for the intermediate store is not empty.

// CHECK: MemorySanitizer: use-of-uninitialized-value
// CHECK:   #0 {{.*}} in main

// CHECK: Uninitialized value was stored to memory at
// CHECK:   #0 {{.*}} in fn_g
// CHECK-NOT: #1

// CHECK: Uninitialized value was created by an allocation of 'z' in the stack frame
// CHECK:   #0 {{.*}} in main

#include <stdio.h>

volatile int x;

__attribute__((noinline))
void fn_g(int a) {
  x = a;
}

__attribute__((noinline))
void fn_f(int a) {
  fn_g(a);
}

int main(int argc, char *argv[]) {
  int volatile z;
  fn_f(z);
  return x;
}
