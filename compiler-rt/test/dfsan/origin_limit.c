// RUN: %clang_dfsan -mllvm -dfsan-track-origins=1 -mllvm -dfsan-fast-16-labels=true %s -o %t
//
// RUN: %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK < %t.out
//
// RUN: DFSAN_OPTIONS=origin_history_size=2 %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK2 < %t.out
//
// RUN: DFSAN_OPTIONS=origin_history_size=0 %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK0 < %t.out

#include <sanitizer/dfsan_interface.h>

#include <stdio.h>

__attribute__((noinline)) int foo(int a, int b) { return a + b; }

int main(int argc, char *argv[]) {
  int a = 10;
  dfsan_set_label(8, &a, sizeof(a));
  int c = 0;
  for (int i = 0; i < 17; ++i) {
    c = foo(a, c);
    printf("%lx", (unsigned long)&c);
  }
  dfsan_print_origin_trace(&c, NULL);
}

// CHECK: Taint value 0x8 {{.*}} origin tracking ()
// CHECK-COUNT 14: Origin value: {{.*}}, Taint value was stored to memory at
// CHECK: Origin value: {{.*}}, Taint value was created at

// CHECK2: Taint value 0x8 {{.*}} origin tracking ()
// CHECK2: Origin value: {{.*}}, Taint value was stored to memory at
// CHECK2: Origin value: {{.*}}, Taint value was created at

// CHECK0: Taint value 0x8 {{.*}} origin tracking ()
// CHECK0-COUNT 16: Origin value: {{.*}}, Taint value was stored to memory at
// CHECK0: Origin value: {{.*}}, Taint value was created at
