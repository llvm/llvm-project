// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -triple s390x-none-zos %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple s390x-linux-gnu %s -verify=loz

typedef struct __attribute__((__aligned__(16))) {
  int a;
  int b;
  int c;
  int d;
} quadruple_word;

int func(quadruple_word *a, quadruple_word *b, quadruple_word *c) {
  return __cdsg(a, b, c);
}
// loz-error@13 {{call to undeclared function '__cdsg'; ISO C99 and later do not support implicit function declarations}}

// CHECK: cmpxchg ptr {{.*}}, i128 {{.*}}, i128 {{.*}} seq_cst seq_cst, align 16
// CHECK: store i128 {{.*}}, ptr {{.*}}, align 16
// CHECK: xor i1 {{.*}}, true
