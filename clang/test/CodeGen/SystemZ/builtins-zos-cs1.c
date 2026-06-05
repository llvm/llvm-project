// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -triple s390x-none-zos %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple s390x-linux-gnu %s -verify

int func(int *a, int *b, int *c) {
  return __cs1(a, b, c);
}
// expected-error@6 {{call to undeclared function '__cs1'; ISO C99 and later do not support implicit function declarations}}

// CHECK: cmpxchg ptr {{.*}}, i32 {{.*}}, i32 {{.*}} seq_cst seq_cst, align 4
// CHECK: store i32 {{.*}}, ptr {{.*}}, align 4
// CHECK: xor i1 {{.*}}, true
