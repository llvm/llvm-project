// RUN: %clang_dfsan -gmlt -DOFFSET=0 -mllvm -dfsan-track-origins=1 %s -o %t && \
// RUN:     %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK0 < %t.out
// 
// RUN: %clang_dfsan -gmlt -DOFFSET=10 -mllvm -dfsan-track-origins=1 %s -o %t && \
// RUN:     %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK10 < %t.out

#include <sanitizer/dfsan_interface.h>

#include <string.h>

int xx[10000];

volatile int idx = 30;

__attribute__((noinline))
void fn_g(int a, int b) {
  memset(&xx[idx], a, sizeof(a));
  memset(&xx[idx + 10], b, sizeof(b));
}

__attribute__((noinline))
void fn_f(int a, int b) {
  fn_g(a, b);
}

int main(int argc, char *argv[]) {
  int volatile z1;
  int volatile z2;
  dfsan_set_label(8, (void *)&z1, sizeof(z1));
  dfsan_set_label(16, (void *)&z2, sizeof(z2));
  fn_f(z1, z2);
  dfsan_print_origin_trace(&xx[idx + OFFSET], NULL);
  return 0;
}

// CHECK0: Taint value 0x8 {{.*}} origin tracking ()
// CHECK0: Origin value: {{.*}}, Taint value was created at

// CHECK0: #0 {{.*}} in main {{.*}}origin_memset.c:[[@LINE-10]]

// CHECK10: Taint value 0x10 {{.*}} origin tracking ()
// CHECK10: Origin value: {{.*}}, Taint value was created at

// CHECK10: #0 {{.*}} in main {{.*}}origin_memset.c:[[@LINE-14]]
