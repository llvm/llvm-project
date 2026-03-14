// RUN: %clang_cc1 -verify -fopenmp --std=c++20 -ast-print %s -Wsign-conversion | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++20 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++20 -include-pch %t -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd --std=c++20 -ast-print %s -Wsign-conversion | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++20 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -std=c++20 -include-pch %t -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

template <typename T> class iterator {
public:
  T &operator*() const;
  iterator &operator++();
};
template <typename T>
bool operator==(const iterator<T> &, const iterator<T> &);
template <typename T>
unsigned long operator-(const iterator<T> &, const iterator<T> &);
template <typename T>
iterator<T> operator+(const iterator<T> &, unsigned long);
class vector {
public:
  vector();
  iterator<int> begin();
  iterator<int> end();
};
// CHECK: void foo() {
void foo() {
// CHECK-NEXT: vector vec;
  vector vec;
// CHECK-NEXT: #pragma omp for
#pragma omp for
// CHECK-NEXT: for (int i : vec)
  for (int i : vec)
    ;
}
#endif
