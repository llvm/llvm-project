// C++ range-for + split: verify syntax, IR, and PreInits (range evaluated once).
//
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -x c++ -std=c++17 -fopenmp -fopenmp-version=60 -fsyntax-only -verify %s
// expected-no-diagnostics
//
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -x c++ -std=c++17 -fopenmp -fopenmp-version=60 -emit-llvm %s -o - | FileCheck %s

extern "C" void body(int);

// CHECK-LABEL: define dso_local void @_Z10range_fillv
// CHECK: __range
// CHECK: __begin
// CHECK: __end
// CHECK: .split.iv.0
// CHECK: icmp slt i64 {{.*}}, 2
// CHECK: call void @body
// CHECK: .split.iv.1
// CHECK: icmp slt
// CHECK: call void @body
void range_fill() {
  int a[] = {10, 20, 30, 40};
#pragma omp split counts(2, omp_fill)
  for (int x : a)
    body(x);
}
