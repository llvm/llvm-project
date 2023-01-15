// RUN: %clang_dfsan -gmlt -mllvm -dfsan-track-origins=1 %s -o %t && \
// RUN:     %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
//
// RUN: %clang_dfsan -gmlt -mllvm -dfsan-track-origins=1 -mllvm -dfsan-instrument-with-call-threshold=0 %s -o %t && \
// RUN:     %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

#include <sanitizer/dfsan_interface.h>

__attribute__((noinline)) uint64_t foo(uint64_t a, uint64_t b) { return a + b; }

int main(int argc, char *argv[]) {
  uint64_t a = 10;
  uint64_t b = 20;
  dfsan_add_label(4, &a, sizeof(a));
  dfsan_add_label(8, &a, sizeof(a));
  uint64_t c = foo(a, b);
  dfsan_print_origin_trace(&c, NULL);
  dfsan_print_origin_trace((int*)&c + 1, NULL);
}

// CHECK: Taint value 0xc {{.*}} origin tracking ()
// CHECK: Origin value: {{.*}}, Taint value was stored to memory at
// CHECK: #0 {{.*}} in main {{.*}}origin_add_label.c:[[@LINE-7]]

// CHECK: Origin value: {{.*}}, Taint value was created at
// CHECK: #0 {{.*}} in main {{.*}}origin_add_label.c:[[@LINE-11]]

// CHECK: Taint value 0xc {{.*}} origin tracking ()
// CHECK: Origin value: {{.*}}, Taint value was stored to memory at
// CHECK: #0 {{.*}} in main {{.*}}origin_add_label.c:[[@LINE-14]]

// CHECK: Origin value: {{.*}}, Taint value was created at
// CHECK: #0 {{.*}} in main {{.*}}origin_add_label.c:[[@LINE-18]]
