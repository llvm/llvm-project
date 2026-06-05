// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -triple s390x-none-zos %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple s390x-linux-gnu %s -verify=loz

typedef struct __attribute__((__aligned__(8))) {
  int a;
  int b;
} double_word;

int func(double_word *a, double_word *b, double_word *c) {
  return __csg(a, b, c);
}
// loz-error@11 {{call to undeclared function '__csg'; ISO C99 and later do not support implicit function declarations}}

// CHECK: cmpxchg ptr {{.*}}, i64 {{.*}}, i64 {{.*}} seq_cst seq_cst, align 8
// CHECK: store i64 {{.*}}, ptr {{.*}}, align 8
// CHECK: xor i1 {{.*}}, true
