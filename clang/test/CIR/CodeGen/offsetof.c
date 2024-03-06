// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %s -o - | FileCheck %s

#include <stddef.h>

typedef struct {
  int a;
  int b;
} A;

void foo() {
  offsetof(A, a);
  offsetof(A, b);
}

// CHECK:  cir.func no_proto @foo()
// CHECK:    {{.*}} = cir.const(#cir.int<0> : !u64i) : !u64i
// CHECK:    {{.*}} = cir.const(#cir.int<4> : !u64i) : !u64i
// CHECK:    cir.return

