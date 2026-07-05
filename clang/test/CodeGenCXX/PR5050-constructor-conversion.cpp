// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -emit-llvm %s -o - | \
// RUN: FileCheck %s
// RUN: %clang_cc1 -triple i386-apple-darwin -std=c++11 -emit-llvm %s -o - | \
// RUN: FileCheck %s

struct A { A(const A&, int i1 = 1); };

struct B : A { };

A f(const B &b) {
  return b;
}

// CHECK: call void @_ZN1AC1ERKS_i
