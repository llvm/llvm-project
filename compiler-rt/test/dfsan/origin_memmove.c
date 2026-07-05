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
  xx[idx] = a; xx[idx + 10] = b;
}

__attribute__((noinline))
void fn_f(int a, int b) {
  fn_g(a, b);
}

__attribute__((noinline))
void fn_h() {
  memmove(&xx[24], &xx, 7500);
}

int main(int argc, char *argv[]) {
  int volatile z1 = 0;
  int volatile z2 = 0;
  dfsan_set_label(8, (void *)&z1, sizeof(z1));
  dfsan_set_label(16, (void *)&z2, sizeof(z2));
  fn_f(z1, z2);
  fn_h();
  dfsan_print_origin_trace(&xx[24 + idx + OFFSET], NULL);
  return 0;
}

// CHECK0: Taint value 0x8 {{.*}} origin tracking ()
// CHECK0: Origin value: {{.*}}, Taint value was stored to memory at
// CHECK0: #0 {{.*}} in fn_h.dfsan {{.*}}origin_memmove.c:[[@LINE-16]]
// CHECK0: #1 {{.*}} in main {{.*}}origin_memmove.c:[[@LINE-8]]

// CHECK0: Origin value: {{.*}}, Taint value was stored to memory at
// CHECK0: #0 {{.*}} in fn_g.dfsan {{.*}}origin_memmove.c:[[@LINE-30]]
// CHECK0: #1 {{.*}} in fn_f.dfsan {{.*}}origin_memmove.c:[[@LINE-26]]
// CHECK0: #2 {{.*}} in main {{.*}}origin_memmove.c:[[@LINE-14]]

// CHECK0: Origin value: {{.*}}, Taint value was created at
// CHECK0: #0 {{.*}} in main {{.*}}origin_memmove.c:[[@LINE-19]]

// CHECK10: Taint value 0x10 {{.*}} origin tracking ()
// CHECK10: Origin value: {{.*}}, Taint value was stored to memory at
// CHECK10: #0 {{.*}} in fn_h.dfsan {{.*}}origin_memmove.c:[[@LINE-29]]
// CHECK10: #1 {{.*}} in main {{.*}}origin_memmove.c:[[@LINE-21]]

// CHECK10: Origin value: {{.*}}, Taint value was stored to memory at
// CHECK10: #0 {{.*}} in fn_g.dfsan {{.*}}origin_memmove.c:[[@LINE-43]]
// CHECK10: #1 {{.*}} in fn_f.dfsan {{.*}}origin_memmove.c:[[@LINE-39]]
// CHECK10: #2 {{.*}} in main {{.*}}origin_memmove.c:[[@LINE-27]]

// CHECK10: Origin value: {{.*}}, Taint value was created at
// CHECK10: #0 {{.*}} in main {{.*}}origin_memmove.c:[[@LINE-31]]
