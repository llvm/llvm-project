// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void foo() {}

template <class T, class U>
T foo(T targ, U uarg) {
  static T a;
  U b;
  int l;
#pragma omp target update to(a) if(l>5) device(l) nowait depend(inout:l)

#pragma omp target update from(b) if(l<5) device(l-1) nowait depend(inout:l)
  return a + targ + (T)b;
}
// CHECK:      static int a;
// CHECK-NEXT: float b;
// CHECK-NEXT: int l;
// CHECK-NEXT: #pragma omp target update to(a) if(l > 5) device(l) nowait depend(inout : l)
// CHECK-NEXT: #pragma omp target update from(b) if(l < 5) device(l - 1) nowait depend(inout : l)
// CHECK:      static char a;
// CHECK-NEXT: float b;
// CHECK-NEXT: int l;
// CHECK-NEXT: #pragma omp target update to(a) if(l > 5) device(l) nowait depend(inout : l)
// CHECK-NEXT: #pragma omp target update from(b) if(l < 5) device(l - 1) nowait depend(inout : l)
// CHECK:      static T a;
// CHECK-NEXT: U b;
// CHECK-NEXT: int l;
// CHECK-NEXT: #pragma omp target update to(a) if(l > 5) device(l) nowait depend(inout : l)
// CHECK-NEXT: #pragma omp target update from(b) if(l < 5) device(l - 1) nowait depend(inout : l)

int main(int argc, char **argv) {
  static int a;
  int n;
  float f;

// CHECK:      static int a;
// CHECK-NEXT: int n;
// CHECK-NEXT: float f;
#pragma omp target update to(a) if(f>0.0) device(n) nowait depend(in:n)
// CHECK-NEXT: #pragma omp target update to(a) if(f > 0.) device(n) nowait depend(in : n)
#pragma omp target update from(f) if(f<0.0) device(n+1) nowait depend(in:n)
// CHECK-NEXT: #pragma omp target update from(f) if(f < 0.) device(n + 1) nowait depend(in : n)
  return foo(argc, f) + foo(argv[0][0], f) + a;
}

#endif
