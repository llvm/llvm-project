// RUN: %clang_dfsan -DTEST64 -DALIGN=8 -mllvm -dfsan-track-origins=1 -mllvm -dfsan-fast-16-labels=true %s -o %t && \
// RUN:     %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK < %t.out
//
// Test origin tracking is accurate in terms of partial store/load, and
// different aligments. Do not test alignments that are not power of 2.
// Compilers do not always allow this.

#include <sanitizer/dfsan_interface.h>

#ifdef TEST64
typedef uint64_t FULL_TYPE;
typedef uint32_t HALF_TYPE;
#elif defined(TEST32)
typedef uint32_t FULL_TYPE;
typedef uint16_t HALF_TYPE;
#else
typedef uint16_t FULL_TYPE;
typedef uint8_t HALF_TYPE;
#endif

__attribute__((noinline)) FULL_TYPE foo(FULL_TYPE a, FULL_TYPE b) { return a + b; }

int main(int argc, char *argv[]) {
  char x __attribute__((aligned(ALIGN))) = 1, y = 2;
  dfsan_set_label(8, &x, sizeof(x));
  char z __attribute__((aligned(ALIGN))) = x + y;
  dfsan_print_origin_trace(&z, NULL);

  FULL_TYPE a __attribute__((aligned(ALIGN))) = 1;
  FULL_TYPE b = 10;
  dfsan_set_label(4, (HALF_TYPE *)&a + 1, sizeof(HALF_TYPE));
  FULL_TYPE c __attribute__((aligned(ALIGN))) = foo(a, b);
  dfsan_print_origin_trace(&c, NULL);
  dfsan_print_origin_trace((HALF_TYPE *)&c + 1, NULL);
}

// CHECK: Taint value 0x8 {{.*}} origin tracking ()
// CHECK: Origin value: {{.*}}, Taint value was stored to memory at

// CHECK: Origin value: {{.*}}, Taint value was created at

// CHECK: Taint value 0x4 {{.*}} origin tracking ()
// CHECK: Origin value: {{.*}}, Taint value was stored to memory at

// CHECK: Origin value: {{.*}}, Taint value was created at

// CHECK: Taint value 0x4 {{.*}} origin tracking ()
// CHECK: Origin value: {{.*}}, Taint value was stored to memory at

// CHECK: Origin value: {{.*}}, Taint value was created at
