// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=50 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=50 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -DOMP51 -verify -fopenmp -fopenmp-version=51 -ast-print %s | FileCheck -check-prefixes=CHECK,OMP51 %s
// RUN: %clang_cc1 -DOMP51 -fopenmp -fopenmp-version=51 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -DOMP51 -fopenmp -fopenmp-version=51 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck -check-prefixes=CHECK,OMP51 %s

// RUN: %clang_cc1 -DOMP51 -verify -fopenmp-simd -fopenmp-version=51 -ast-print %s | FileCheck -check-prefixes=CHECK,OMP51 %s
// RUN: %clang_cc1 -DOMP51 -fopenmp-simd -fopenmp-version=51 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -DOMP51 -fopenmp-simd -fopenmp-version=51 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck -check-prefixes=CHECK,OMP51 %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void foo() {}

template <class T, class U>
T foo(T targ, U uarg) {
  static T a, *p;
  U b;
  int l;
#pragma omp target update to(([a][targ])p, a) if(l>5) device(l) nowait depend(inout:l)

#pragma omp target update from(b, ([a][targ])p) if(l<5) device(l-1) nowait depend(inout:l)

  int arr[100][100];
#pragma omp target update to(arr[2][0:1:2])

#pragma omp target update from(arr[2][0:1:2])

#ifdef OMP51
#pragma omp target update to(present: arr[2][0:1:2])

#pragma omp target update from(present: arr[2][0:1:2], a)
#endif

  return a + targ + (T)b;
}
// CHECK:      static T a, *p;
// CHECK-NEXT: U b;
// CHECK-NEXT: int l;
// CHECK-NEXT: #pragma omp target update to(([a][targ])p,a) if(l > 5) device(l) nowait depend(inout : l){{$}}
// CHECK-NEXT: #pragma omp target update from(b,([a][targ])p) if(l < 5) device(l - 1) nowait depend(inout : l)
// CHECK:      static int a, *p;
// CHECK-NEXT: float b;
// CHECK-NEXT: int l;
// CHECK-NEXT: #pragma omp target update to(([a][targ])p,a) if(l > 5) device(l) nowait depend(inout : l)
// CHECK-NEXT: #pragma omp target update from(b,([a][targ])p) if(l < 5) device(l - 1) nowait depend(inout : l)
// CHECK:      static char a, *p;
// CHECK-NEXT: float b;
// CHECK-NEXT: int l;
// CHECK-NEXT: #pragma omp target update to(([a][targ])p,a) if(l > 5) device(l) nowait depend(inout : l)
// CHECK-NEXT: #pragma omp target update from(b,([a][targ])p) if(l < 5) device(l - 1) nowait depend(inout : l)
// CHECK:      int arr[100][100];
// CHECK-NEXT: #pragma omp target update to(arr[2][0:1:2])
// CHECK-NEXT: #pragma omp target update from(arr[2][0:1:2])
// OMP5-NEXT: #pragma omp target update to(present: arr[2][0:1:2])
// OMP5-NEXT: #pragma omp target update from(present: arr[2][0:1:2], a)

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
#pragma omp target update to(argv[2][0:1:2])
// CHECK-NEXT: #pragma omp target update to(argv[2][0:1:2])
#pragma omp target update from(argv[2][0:1:2])
// CHECK-NEXT: #pragma omp target update from(argv[2][0:1:2])
#ifdef OMP51
#pragma omp target update to(present: argv[2][0:1:2])
// OMP5-NEXT: #pragma omp target update to(present: arr[2][0:1:2])
#pragma omp target update from(argv[2][0:1:2], a)
// OMP5-NEXT: #pragma omp target update from(present: arr[2][0:1:2], a)
#endif


  return foo(argc, f) + foo(argv[0][0], f) + a;
}

#endif
