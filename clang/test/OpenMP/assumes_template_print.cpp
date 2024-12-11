// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -std=c++11 -include-pch %t -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

// It is unclear if we want to annotate the template instantiations, e.g., S<int>::foo, or not in the two
// situations shown below. Since it is always fair to drop assumptions, we do that for now.

#ifndef HEADER
#define HEADER

template <typename T>
struct S {
  int a;
// CHECK: template <typename T> struct S {
// CHECK{LITERAL}:     void foo() [[omp::assume("ompx_global_assumption")]] {
  void foo() {
    #pragma omp parallel
    {}
  }
};

// CHECK: template<> struct S<int> {
// CHECK{LITERAL}:     void foo() [[omp::assume("ompx_global_assumption")]] {

#pragma omp begin assumes no_openmp
// CHECK{LITERAL}: [[omp::assume("omp_no_openmp")]] void S_with_assumes_no_call() [[omp::assume("ompx_global_assumption")]] {
void S_with_assumes_no_call() {
  S<int> s;
  s.a = 0;
}
// CHECK{LITERAL}: [[omp::assume("omp_no_openmp")]] void S_with_assumes_call() [[omp::assume("ompx_global_assumption")]] {
void S_with_assumes_call() {
  S<int> s;
  s.a = 0;
  // If this is executed we have UB!
  s.foo();
}
#pragma omp end assumes

// CHECK{LITERAL}: void S_without_assumes() [[omp::assume("ompx_global_assumption")]] {
void S_without_assumes() {
  S<int> s;
  s.foo();
}

#pragma omp assumes ext_global_assumption

// Same as the struct S above but the order in which we instantiate P is different, first outside of an assumes.
template <typename T>
struct P {
// CHECK: template <typename T> struct P {
// CHECK{LITERAL}:    [[omp::assume("ompx_global_assumption")]] void foo() {
  int a;
  void foo() {
    #pragma omp parallel
    {}
  }
};

// TODO: Avoid the duplication here:

// CHECK: template<> struct P<int> {
// CHECK{LITERAL}:     [[omp::assume("ompx_global_assumption")]] [[omp::assume("ompx_global_assumption")]] void foo() {

// CHECK{LITERAL}: [[omp::assume("ompx_global_assumption")]] void P_without_assumes() {
void P_without_assumes() {
  P<int> p;
  p.foo();
}

#pragma omp begin assumes no_openmp
// CHECK{LITERAL}: [[omp::assume("omp_no_openmp")]] [[omp::assume("ompx_global_assumption")]] void P_with_assumes_no_call() {
void P_with_assumes_no_call() {
  P<int> p;
  p.a = 0;
}
// CHECK{LITERAL}: [[omp::assume("omp_no_openmp")]] [[omp::assume("ompx_global_assumption")]] void P_with_assumes_call() {
void P_with_assumes_call() {
  P<int> p;
  p.a = 0;
  // If this is executed we have UB!
  p.foo();
}
#pragma omp end assumes

#endif
