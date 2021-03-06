// RUN: %clang_dfsan -gmlt -DOFFSET=0 -mllvm -dfsan-track-origins=1 -mllvm -dfsan-fast-16-labels=true %s -o %t && \
// RUN:     %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK0 < %t.out

// RUN: %clang_dfsan -gmlt -DOFFSET=10 -mllvm -dfsan-track-origins=1 -mllvm -dfsan-fast-16-labels=true %s -o %t && \
// RUN:     %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK10 < %t.out

#include <sanitizer/dfsan_interface.h>

#include <string.h>

char xx[10000];
char yy[10000];
volatile int idx = 30;

__attribute__((noinline)) void fn_g(char a, char b) {
  xx[idx] = a; xx[idx + 10] = b;
}

__attribute__((noinline)) void fn_f(char a, char b) {
  fn_g(a, b);
}

__attribute__((noinline)) void fn_h() {
  memcpy(&yy[2], &xx[2], sizeof(xx) - 4);
}

__attribute__((noinline)) void fn_i() {
  memmove(&yy[25], &yy, 7500);
}

int main(int argc, char *argv[]) {
  char volatile z1 = 0;
  int volatile buffer = 0;
  char volatile z2 = 0;
  dfsan_set_label(8, (void *)&z1, sizeof(z1));
  dfsan_set_label(16, (void *)&z2, sizeof(z2));
  fn_f(z1, z2);
  fn_h();
  fn_i();
  dfsan_print_origin_trace(&yy[25 + idx + OFFSET], NULL);
  return 0;
}

// CHECK0: Taint value 0x8 {{.*}} origin tracking ()
// CHECK0: Origin value: {{.*}}, Taint value was stored to memory at
// CHECK0: #0 {{.*}} in dfs$fn_i {{.*}}origin_unaligned_memtrans.c:[[@LINE-18]]
// CHECK0: #1 {{.*}} in main {{.*}}origin_unaligned_memtrans.c:[[@LINE-8]]

// CHECK0: Origin value: {{.*}}, Taint value was stored to memory at
// CHECK0: #0 {{.*}} in dfs$fn_h {{.*}}origin_unaligned_memtrans.c:[[@LINE-26]]
// CHECK0: #1 {{.*}} in main {{.*}}origin_unaligned_memtrans.c:[[@LINE-13]]

// CHECK0: Origin value: {{.*}}, Taint value was stored to memory at
// CHECK0: #0 {{.*}} in dfs$fn_g {{.*}}origin_unaligned_memtrans.c:[[@LINE-38]]
// CHECK0: #1 {{.*}} in dfs$fn_f {{.*}}origin_unaligned_memtrans.c:[[@LINE-35]]
// CHECK0: #2 {{.*}} in main {{.*}}origin_unaligned_memtrans.c:[[@LINE-19]]

// CHECK0: Origin value: {{.*}}, Taint value was created at
// CHECK0: #0 {{.*}} in main {{.*}}origin_unaligned_memtrans.c:[[@LINE-24]]

// CHECK10: Taint value 0x10 {{.*}} origin tracking
// CHECK10: Origin value: {{.*}}, Taint value was stored to memory at
// CHECK10: #0 {{.*}} in dfs$fn_i {{.*}}origin_unaligned_memtrans.c:[[@LINE-35]]
// CHECK10: #1 {{.*}} in main {{.*}}origin_unaligned_memtrans.c:[[@LINE-25]]

// CHECK10: Origin value: {{.*}}, Taint value was stored to memory at
// CHECK10: #0 {{.*}} in dfs$fn_h {{.*}}origin_unaligned_memtrans.c:[[@LINE-43]]
// CHECK10: #1 {{.*}} in main {{.*}}origin_unaligned_memtrans.c:[[@LINE-30]]

// CHECK10: Origin value: {{.*}}, Taint value was stored to memory at
// CHECK10: #0 {{.*}} in dfs$fn_g {{.*}}origin_unaligned_memtrans.c:[[@LINE-55]]
// CHECK10: #1 {{.*}} in dfs$fn_f {{.*}}origin_unaligned_memtrans.c:[[@LINE-52]]
// CHECK10: #2 {{.*}} in main {{.*}}origin_unaligned_memtrans.c:[[@LINE-36]]

// CHECK10: Origin value: {{.*}}, Taint value was created at
// CHECK10: #0 {{.*}} in main {{.*}}origin_unaligned_memtrans.c:[[@LINE-40]]
