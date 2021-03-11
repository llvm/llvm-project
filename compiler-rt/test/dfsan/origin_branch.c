// RUN: %clang_dfsan -gmlt -mllvm -dfsan-track-origins=1 -mllvm -dfsan-fast-16-labels=true %s -o %t && \
// RUN:     %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK < %t.out

#include <sanitizer/dfsan_interface.h>

__attribute__((noinline)) uint64_t foo(uint64_t x, uint64_t y) { return x + y; }

int main(int argc, char *argv[]) {
  uint64_t a = 10;
  uint64_t b = 20;
  dfsan_set_label(8, &a, sizeof(a));
  uint64_t c = foo(a, b);
  for (int i = 0; i < argc; ++i)
    c += foo(c, b);
  dfsan_print_origin_trace(&c, NULL);
}

// CHECK: Taint value 0x8 {{.*}} origin tracking ()
// CHECK: Origin value: {{.*}}, Taint value was stored to memory at
// CHECK: #0 {{.*}} in main {{.*}}origin_branch.c:[[@LINE-6]]

// CHECK: Origin value: {{.*}}, Taint value was stored to memory at
// CHECK: #0 {{.*}} in main {{.*}}origin_branch.c:[[@LINE-11]]

// CHECK: Origin value: {{.*}}, Taint value was created at
// CHECK: #0 {{.*}} in main {{.*}}origin_branch.c:[[@LINE-15]]
