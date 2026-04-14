/* C++ `constexpr` locals as `counts` operands (distinct from NTTP in split_template_nttp.cpp). */
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -x c++ -std=c++17 -fopenmp -fopenmp-version=60 -fsyntax-only -verify %s
// expected-no-diagnostics
//
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -x c++ -std=c++17 -fopenmp -fopenmp-version=60 -O0 -emit-llvm %s -o - | FileCheck %s

extern "C" void body(int);

// CHECK-LABEL: define {{.*}} @from_constexpr
// CHECK: .split.iv.0
// CHECK: icmp slt i32 {{.*}}, 4
// CHECK: .split.iv.1
// CHECK: icmp slt i32 {{.*}}, 10
extern "C" void from_constexpr(void) {
  static constexpr int C0 = 4;
#pragma omp split counts(C0, omp_fill)
  for (int i = 0; i < 10; ++i)
    body(i);
}
