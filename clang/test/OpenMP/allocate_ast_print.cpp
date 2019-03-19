// RUN: %clang_cc1 -verify -fopenmp -triple x86_64-apple-darwin10.6.0 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -triple x86_64-apple-darwin10.6.0 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -triple x86_64-apple-darwin10.6.0 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print
// RUN: %clang_cc1 -verify -fopenmp -triple x86_64-unknown-linux-gnu -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fnoopenmp-use-tls -triple x86_64-unknown-linux-gnu -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fnoopenmp-use-tls -triple x86_64-unknown-linux-gnu -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print

// RUN: %clang_cc1 -verify -fopenmp-simd -triple x86_64-apple-darwin10.6.0 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -triple x86_64-apple-darwin10.6.0 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -triple x86_64-apple-darwin10.6.0 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print
// RUN: %clang_cc1 -verify -fopenmp-simd -triple x86_64-unknown-linux-gnu -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -fnoopenmp-use-tls -triple x86_64-unknown-linux-gnu -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fnoopenmp-use-tls -triple x86_64-unknown-linux-gnu -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

struct St{
 int a;
};

struct St1{
 int a;
 static int b;
// CHECK: static int b;
#pragma omp allocate(b)
// CHECK-NEXT: #pragma omp allocate(St1::b){{$}}
} d;

int a, b;
// CHECK: int a;
// CHECK: int b;
#pragma omp allocate(a)
#pragma omp allocate(a)
// CHECK-NEXT: #pragma omp allocate(a)
// CHECK-NEXT: #pragma omp allocate(a)
#pragma omp allocate(d, b)
// CHECK-NEXT: #pragma omp allocate(d,b)

template <class T>
struct ST {
  static T m;
  #pragma omp allocate(m)
};

template <class T> T foo() {
  T v;
  #pragma omp allocate(v)
  v = ST<T>::m;
  return v;
}
//CHECK: template <class T> T foo() {
//CHECK-NEXT: T v;
//CHECK-NEXT: #pragma omp allocate(v)
//CHECK: template<> int foo<int>() {
//CHECK-NEXT: int v;
//CHECK-NEXT: #pragma omp allocate(v)

namespace ns{
  int a;
}
// CHECK: namespace ns {
// CHECK-NEXT: int a;
// CHECK-NEXT: }
#pragma omp allocate(ns::a)
// CHECK-NEXT: #pragma omp allocate(ns::a)

int main () {
  static int a;
// CHECK: static int a;
#pragma omp allocate(a)
// CHECK-NEXT: #pragma omp allocate(a)
  a=2;
  return (foo<int>());
}

extern template int ST<int>::m;
#endif
