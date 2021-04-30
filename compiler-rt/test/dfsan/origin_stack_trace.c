// RUN: %clang_dfsan -gmlt -mllvm -dfsan-track-origins=1 -mllvm -dfsan-fast-16-labels=true %s -o %t && \
// RUN:     %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK < %t.out
//
// RUN: %clang_dfsan -gmlt -mllvm -dfsan-track-origins=1 -mllvm -dfsan-fast-16-labels=true -mllvm -dfsan-instrument-with-call-threshold=0 %s -o %t && \
// RUN:     %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK < %t.out
//
// REQUIRES: x86_64-target-arch

#include <sanitizer/dfsan_interface.h>

#define NOINLINE __attribute__((noinline))

NOINLINE int foo(int a, int b) { return a + b; }

NOINLINE void bar(int depth, void *addr, int size) {
  if (depth) {
    bar(depth - 1, addr, size);
  } else {
    dfsan_set_label(1, addr, size);
  }
}

NOINLINE void baz(int depth, void *addr, int size) {
  bar(depth, addr, size);
}

int main(int argc, char *argv[]) {
  int a = 10;
  int b = 20;
  baz(8, &a, sizeof(a));
  int c = foo(a, b);
  dfsan_print_origin_trace(&c, NULL);
}

// CHECK: Taint value 0x1 {{.*}} origin tracking ()
// CHECK: Origin value: {{.*}}, Taint value was stored to memory at
// CHECK: #0 {{.*}} in main {{.*}}origin_stack_trace.c:[[@LINE-6]]

// CHECK: Origin value: {{.*}}, Taint value was created at
// CHECK: #0 {{.*}} in dfs$bar {{.*}}origin_stack_trace.c:[[@LINE-21]]
// CHECK-COUNT-8: #{{[0-9]+}} {{.*}} in dfs$bar {{.*}}origin_stack_trace.c:[[@LINE-24]]
// CHECK: #9 {{.*}} in dfs$baz {{.*}}origin_stack_trace.c:[[@LINE-18]]
