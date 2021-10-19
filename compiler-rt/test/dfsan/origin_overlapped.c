// RUN: %clang_dfsan -gmlt -mllvm -dfsan-track-origins=1 %s -o %t && \
// RUN:     %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
//
// REQUIRES: x86_64-target-arch

#include <sanitizer/dfsan_interface.h>

int main(int argc, char *argv[]) {
  char volatile z1;
  char volatile z2;
  dfsan_set_label(8, (void *)&z1, sizeof(z1));
  dfsan_set_label(16, (void *)&z2, sizeof(z2)); // overwritting the old origin.
  char c = z1;
  dfsan_print_origin_trace(&c, "bar");
  return 0;
}

// CHECK: Taint value 0x8 {{.*}} origin tracking (bar)
// CHECK: Origin value: {{.*}}, Taint value was stored to memory at
// CHECK: #0 {{.*}} in main {{.*}}origin_overlapped.c:[[@LINE-7]]

// CHECK: Origin value: {{.*}}, Taint value was created at

// CHECK: #0 {{.*}} in main {{.*}}origin_overlapped.c:[[@LINE-12]]
