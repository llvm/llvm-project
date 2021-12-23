// RUN: %clang_dfsan -gmlt -mllvm -dfsan-track-origins=1 %s -o %t && \
// RUN:     %run %t 2>&1 | FileCheck %s
//
// REQUIRES: x86_64-target-arch

#include <assert.h>
#include <sanitizer/dfsan_interface.h>
#include <stdio.h>

__attribute__((noinline)) uint64_t foo(uint64_t a, uint64_t b) { return a + b; }

int main(int argc, char *argv[]) {
  uint64_t a = 10;
  uint64_t b = 20;
  dfsan_set_label(8, &a, sizeof(a));
  uint64_t c = foo(a, b);

  dfsan_origin c_orig = dfsan_get_origin(c);
  fprintf(stderr, "c_orig 0x%x\n", c_orig);
  // CHECK: c_orig 0x[[#%x,C_ORIG:]]
  assert(c_orig != 0);
  dfsan_print_origin_id_trace(c_orig);
  // CHECK: Origin value: 0x[[#%x,C_ORIG]], Taint value was created at

  uint64_t d[4] = {1, 2, 3, c};
  dfsan_origin d_orig = dfsan_read_origin_of_first_taint(d, sizeof(d));
  fprintf(stderr, "d_orig 0x%x\n", d_orig);
  // CHECK: d_orig 0x[[#%x,D_ORIG:]]
  assert(d_orig != 0);
  dfsan_print_origin_id_trace(d_orig);
  // CHECK: Origin value: 0x[[#%x,D_ORIG]], Taint value was stored to memory at
  // CHECK: Origin value: 0x[[#%x,C_ORIG]], Taint value was created at
  return 0;
}
