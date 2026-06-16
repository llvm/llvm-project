// RUN: %clang_dfsan -gmlt -mllvm -dfsan-track-origins=1 %s -o %t && \
// RUN:     %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
//
// Test origin tracking is accurate in terms of endianness.

#include <sanitizer/dfsan_interface.h>

typedef uint64_t FULL_TYPE;
typedef uint32_t HALF_TYPE;

__attribute__((noinline)) FULL_TYPE foo(FULL_TYPE a, FULL_TYPE b) {
  return a + b;
}

int main(int argc, char *argv[]) {
  FULL_TYPE a = 1;
  FULL_TYPE b = 10;
  dfsan_set_label(4, (HALF_TYPE *)&a + 1, sizeof(HALF_TYPE));
  FULL_TYPE c = foo(a, b);
  dfsan_print_origin_trace(&c, NULL);
  dfsan_print_origin_trace((HALF_TYPE *)&c + 1, NULL);
}

// CHECK: Taint value 0x4 {{.*}} origin tracking ()
// CHECK: Origin value: {{.*}}, Taint value was stored to memory at
// CHECK: #0 {{.*}} in main {{.*}}origin_endianness.c:[[@LINE-7]]

// CHECK: Origin value: {{.*}}, Taint value was created at
// CHECK: #0 {{.*}} in main {{.*}}origin_endianness.c:[[@LINE-11]]

// CHECK: Taint value 0x4 {{.*}} origin tracking ()
// CHECK: Origin value: {{.*}}, Taint value was stored to memory at
// CHECK: #0 {{.*}} in main {{.*}}origin_endianness.c:[[@LINE-14]]

// CHECK: Origin value: {{.*}}, Taint value was created at
// CHECK: #0 {{.*}} in main {{.*}}origin_endianness.c:[[@LINE-18]]
