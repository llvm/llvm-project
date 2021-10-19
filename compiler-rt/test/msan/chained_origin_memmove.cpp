// RUN: %clangxx_msan -fsanitize-memory-track-origins=2 -DOFFSET=0 -O3 %s -o %t && \
// RUN:     not %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-Z1 --check-prefix=CHECK-%short-stack < %t.out

// RUN: %clangxx_msan -fsanitize-memory-track-origins=2 -DOFFSET=10 -O3 %s -o %t && \
// RUN:     not %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-Z2 --check-prefix=CHECK-%short-stack < %t.out

// RUN: %clangxx_msan -mllvm -msan-instrumentation-with-call-threshold=0 -fsanitize-memory-track-origins=2 -DOFFSET=0 -O3 %s -o %t && \
// RUN:     not %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-Z1 --check-prefix=CHECK-%short-stack < %t.out

// RUN: %clangxx_msan -mllvm -msan-instrumentation-with-call-threshold=0 -fsanitize-memory-track-origins=2 -DOFFSET=10 -O3 %s -o %t && \
// RUN:     not %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-Z2 --check-prefix=CHECK-%short-stack < %t.out

#include <stdio.h>
#include <string.h>

int xx[10000];
volatile int idx = 30;

__attribute__((noinline)) void fn_g(int a, int b) {
  xx[idx + OFFSET] = OFFSET == 0 ? a : b;
}

__attribute__((noinline)) void fn_f(int a, int b) {
  fn_g(a, b);
}

__attribute__((noinline)) void fn_h() {
  memmove(&xx[25], &xx, 7500);
}

int main(int argc, char *argv[]) {
  int volatile z1;
  int volatile z2;
  fn_f(z1, z2);
  fn_h();
  return xx[25 + idx + OFFSET];
}

// CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
// CHECK: {{#0 .* in main .*chained_origin_memmove.cpp:}}[[@LINE-4]]

// CHECK: Uninitialized value was stored to memory at
// CHECK-FULL-STACK: {{#1 .* in fn_h.*chained_origin_memmove.cpp:}}[[@LINE-15]]
// CHECK-SHORT-STACK: {{#0 .* in __msan_memmove.*msan_interceptors.cpp:}}

// CHECK: Uninitialized value was stored to memory at
// CHECK-FULL-STACK: {{#0 .* in fn_g.*chained_origin_memmove.cpp:}}[[@LINE-27]]
// CHECK-FULL-STACK: {{#1 .* in fn_f.*chained_origin_memmove.cpp:}}[[@LINE-24]]
// CHECK-SHORT-STACK: {{#0 .* in fn_g.*chained_origin_memmove.cpp:}}[[@LINE-29]]

// CHECK-Z1: Uninitialized value was created by an allocation of 'z1' in the stack frame of function 'main'
// CHECK-Z2: Uninitialized value was created by an allocation of 'z2' in the stack frame of function 'main'
// CHECK: {{#0 .* in main.*chained_origin_memmove.cpp:}}[[@LINE-22]]
