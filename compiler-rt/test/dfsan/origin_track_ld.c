// RUN: %clang_dfsan -gmlt -mllvm -dfsan-track-origins=2 %s -o %t && \
// RUN:     %run %t > %t.out 2>&1
// RUN: FileCheck %s < %t.out
//
// REQUIRES: x86_64-target-arch

#include <sanitizer/dfsan_interface.h>

__attribute__((noinline)) uint64_t foo(uint64_t a, uint64_t b) { return a + b; }

int main(int argc, char *argv[]) {
  uint64_t a = 10;
  uint64_t b = 20;
  dfsan_set_label(8, &a, sizeof(a));
  uint64_t c = foo(a, b);
  dfsan_print_origin_trace(&c, NULL);
}

// CHECK: Taint value 0x8 {{.*}} origin tracking ()
// CHECK: Origin value: {{.*}}, Taint value was stored to memory at
// CHECK: #0 {{.*}} in main {{.*}}origin_track_ld.c:[[@LINE-6]]

// CHECK: Origin value: {{.*}}, Taint value was stored to memory at
// CHECK: #0 {{.*}} in foo.dfsan {{.*}}origin_track_ld.c:[[@LINE-15]]
// CHECK: #1 {{.*}} in main {{.*}}origin_track_ld.c:[[@LINE-10]]

// CHECK: Origin value: {{.*}}, Taint value was stored to memory at
// CHECK: #0 {{.*}} in main {{.*}}origin_track_ld.c:[[@LINE-13]]

// CHECK: Origin value: {{.*}}, Taint value was created at
// CHECK: #0 {{.*}} in main {{.*}}origin_track_ld.c:[[@LINE-17]]
