// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -std=c++11 -include-pch %t -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

template <typename> class vector {
public:
  int &at(long);
};

// CHECK:      template <typename T> void foo(int n) {
// CHECK-NEXT:   vector<T> v;
// CHECK-NEXT:   vector<int> iv1;
// CHECK-NEXT: #pragma omp task depend(iterator(int i = 0:n), in : v.at(i),iv1.at(i))
// CHECK-NEXT:     ;
// CHECK-NEXT: }

template <typename T> void foo(int n) {
  vector<T> v;
  vector<int> iv1;
#pragma omp task depend(iterator(i = 0 : n), in : v.at(i), iv1.at(i))
  ;
}
#endif
