// Non-type template parameter as counts operand — IR after instantiation.
//
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -x c++ -std=c++17 -fopenmp -fopenmp-version=60 -O0 -emit-llvm %s -o - | FileCheck %s

// CHECK-LABEL: define {{.*}} @_Z1fILi5EEvv
// CHECK: .split.iv
// CHECK: icmp slt i32{{.*}} 5
template <int N>
void f() {
#pragma omp split counts(N, omp_fill)
  for (int i = 0; i < 20; ++i) {
  }
}

template void f<5>();
