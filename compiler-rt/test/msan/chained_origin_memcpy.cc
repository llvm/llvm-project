// RUN: %clangxx_msan -fsanitize-memory-track-origins=2 -DOFFSET=0 -O3 %s -o %t && \
// RUN:     not %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-Z1 < %t.out

// RUN: %clangxx_msan -fsanitize-memory-track-origins=2 -DOFFSET=10 -O3 %s -o %t && \
// RUN:     not %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-Z2 < %t.out


// RUN: %clangxx_msan -mllvm -msan-instrumentation-with-call-threshold=0 -fsanitize-memory-track-origins=2 -DOFFSET=0 -O3 %s -o %t && \
// RUN:     not %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-Z1 < %t.out

// RUN: %clangxx_msan -mllvm -msan-instrumentation-with-call-threshold=0 -fsanitize-memory-track-origins=2 -DOFFSET=10 -O3 %s -o %t && \
// RUN:     not %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-Z2 < %t.out

#include <stdio.h>
#include <string.h>

int xx[10000];
int yy[10000];
volatile int idx = 30;

__attribute__((noinline))
void fn_g(int a, int b) {
  xx[idx] = a; xx[idx + 10] = b;
}

__attribute__((noinline))
void fn_f(int a, int b) {
  fn_g(a, b);
}

__attribute__((noinline))
void fn_h() {
  memcpy(&yy, &xx, sizeof(xx));
}

int main(int argc, char *argv[]) {
  int volatile z1;
  int volatile z2;
  fn_f(z1, z2);
  fn_h();
  return yy[idx + OFFSET];
}

// CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
// CHECK: {{#0 .* in main .*chained_origin_memcpy.cc:}}[[@LINE-4]]

// CHECK: Uninitialized value was stored to memory at
// CHECK: {{#1 .* in fn_h.*chained_origin_memcpy.cc:}}[[@LINE-15]]

// CHECK: Uninitialized value was stored to memory at
// CHECK: {{#0 .* in fn_g.*chained_origin_memcpy.cc:}}[[@LINE-28]]
// CHECK: {{#1 .* in fn_f.*chained_origin_memcpy.cc:}}[[@LINE-24]]

// CHECK-Z1: Uninitialized value was created by an allocation of 'z1' in the stack frame of function 'main'
// CHECK-Z2: Uninitialized value was created by an allocation of 'z2' in the stack frame of function 'main'
// CHECK: {{#0 .* in main.*chained_origin_memcpy.cc:}}[[@LINE-20]]
