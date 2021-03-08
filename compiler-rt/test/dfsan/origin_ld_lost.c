// RUN: %clang_dfsan -gmlt -mllvm -dfsan-track-origins=1 -mllvm -dfsan-fast-16-labels=true %s -o %t && \
// RUN:     %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK < %t.out
//
// Test origin tracking can lost origins at 2-byte load with addr % 4 == 3.

#include <sanitizer/dfsan_interface.h>

__attribute__((noinline)) uint16_t foo(uint16_t a, uint16_t b) { return a + b; }

int main(int argc, char *argv[]) {
  uint64_t a __attribute__((aligned(4))) = 1;
  uint32_t b = 10;
  dfsan_set_label(4, (uint8_t *)&a + 4, sizeof(uint8_t));
  uint16_t c = foo(*(uint16_t *)((uint8_t *)&a + 3), b);
  dfsan_print_origin_trace(&c, "foo");
}

// CHECK: Taint value 0x4 {{.*}} origin tracking (foo)
// CHECK: Origin value: {{.*}}, Taint value was created at
// CHECK: #0 {{.*}} in main {{.*}}origin_ld_lost.c:[[@LINE-6]]
