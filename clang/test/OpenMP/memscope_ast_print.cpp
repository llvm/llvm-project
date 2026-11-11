// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -fopenmp-version=60 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -fopenmp-version=60 -include-pch %t -verify %s -ast-print | FileCheck %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -fopenmp-version=60 -include-pch %t -ast-dump-all %s | FileCheck %s --check-prefix=DUMP
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

template <class T> T tmain(T x) {
  T v;
#pragma omp flush memscope(all)
#pragma omp flush release memscope(cgroup)
#pragma omp atomic read acquire memscope(device)
  v = x;
  return v;
}

// CHECK-LABEL: template <class T> T tmain(T x) {
// CHECK: #pragma omp flush memscope(all)
// CHECK-NEXT: #pragma omp flush release memscope(cgroup)
// CHECK-NEXT: #pragma omp atomic read acquire memscope(device)
// CHECK-NEXT: v = x;

// CHECK-LABEL: template<> int tmain<int>(int x) {
// CHECK: #pragma omp flush memscope(all)
// CHECK-NEXT: #pragma omp flush release memscope(cgroup)
// CHECK-NEXT: #pragma omp atomic read acquire memscope(device)
// CHECK-NEXT: v = x;

// DUMP-LABEL: FunctionDecl {{.*}} tmain 'int (int)'
// DUMP: OMPFlushDirective
// DUMP-NEXT: OMPMemscopeClause
// DUMP: OMPAtomicDirective
// DUMP-NEXT: OMPReadClause
// DUMP-NEXT: OMPAcquireClause
// DUMP-NEXT: OMPMemscopeClause

int main() {
  int x = 0;
  return tmain(x);
}

#endif
