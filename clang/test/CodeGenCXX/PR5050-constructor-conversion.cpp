// REQUIRES: x86-registered-target,x86-64-registered-target
// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -S %s -o %t-64.s
// RUN: FileCheck -check-prefix LP64 --input-file=%t-64.s %s
// RUN: %clang_cc1 -triple i386-apple-darwin -std=c++11 -S %s -o %t-32.s
// RUN: FileCheck -check-prefix LP32 --input-file=%t-32.s %s

struct A { A(const A&, int i1 = 1); };

struct B : A { };

A f(const B &b) {
  return b;
}

// CHECK-LP64: callq    __ZN1AC1ERKS_i

// CHECK-LP32: calll     L__ZN1AC1ERKS_i


