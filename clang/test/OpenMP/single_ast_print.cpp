// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void foo() {}

struct SS {
  int a;
  int b : 4;
  int &c;
  SS(int &d) : a(0), b(0), c(d) {
#pragma omp parallel firstprivate(a, b, c)
#pragma omp single copyprivate(a, this->b, (this)->c)
// CHECK: #pragma omp parallel firstprivate(this->a,this->b,this->c)
// CHECK-NEXT: #pragma omp single copyprivate(this->a,this->b,this->c){{$}}
    ++this->a, --b, (this)->c /= 1;
  }
};

template<typename T>
struct SST {
  T a;
  SST() : a(T()) {
// CHECK: #pragma omp parallel firstprivate(this->a)
// CHECK-NEXT: #pragma omp single copyprivate(this->a)
// CHECK: #pragma omp parallel firstprivate(this->a)
// CHECK-NEXT: #pragma omp single copyprivate(this->a)
#pragma omp parallel firstprivate(a)
#pragma omp single copyprivate(this->a)
    ++this->a;
  }
};

template <class T, int N>
T tmain(T argc) {
  T b = argc, c, d, e, f, g;
  static T a;
  SST<T> sst;
// CHECK: static T a;
#pragma omp parallel private(g)
#pragma omp single private(argc, b), firstprivate(c, d), nowait
  foo();
  // CHECK: #pragma omp parallel private(g)
  // CHECK-NEXT: #pragma omp single private(argc,b) firstprivate(c,d) nowait
  // CHECK-NEXT: foo();
#pragma omp parallel private(g)
#pragma omp single private(argc, b), firstprivate(c, d), copyprivate(g)
  foo();
  // CHECK-NEXT: #pragma omp parallel private(g)
  // CHECK-NEXT: #pragma omp single private(argc,b) firstprivate(c,d) copyprivate(g)
  // CHECK-NEXT: foo();
  return T();
}

int main(int argc, char **argv) {
// CHECK: int main(int argc, char **argv) {
  int b = argc, c, d, e, f, g;
  static int a;
  SS ss(a);
// CHECK: static int a;
#pragma omp parallel private(g)
#pragma omp single private(argc, b), firstprivate(argv, c), nowait
  foo();
  // CHECK: #pragma omp parallel private(g)
  // CHECK-NEXT: #pragma omp single private(argc,b) firstprivate(argv,c) nowait
  // CHECK-NEXT: foo();
#pragma omp parallel private(g)
#pragma omp single private(argc, b), firstprivate(c, d), copyprivate(g)
  foo();
  // CHECK-NEXT: #pragma omp parallel private(g)
  // CHECK-NEXT: #pragma omp single private(argc,b) firstprivate(c,d) copyprivate(g)
  // CHECK-NEXT: foo();
  return (tmain<int, 5>(argc) + tmain<char, 1>(argv[0][0]));
}

#endif
