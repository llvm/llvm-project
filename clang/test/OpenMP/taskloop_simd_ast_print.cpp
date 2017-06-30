// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void foo() {}

template <class T, int N>
T tmain(T argc) {
  T b = argc, c, d, e, f, g;
  T *ptr;
  static T a;
// CHECK: static T a;
#pragma omp taskloop simd if(taskloop: argc > N) default(shared) untied priority(N) safelen(N) linear(c) aligned(ptr) grainsize(N) reduction(+:g)
  // CHECK-NEXT: #pragma omp taskloop simd if(taskloop: argc > N) default(shared) untied priority(N) safelen(N) linear(c) aligned(ptr) grainsize(N) reduction(+: g)
  for (int i = 0; i < 2; ++i)
    a = 2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp parallel
#pragma omp taskloop simd private(argc, b), firstprivate(c, d), lastprivate(d, f) collapse(N) shared(g) if (c) final(d) mergeable priority(f) simdlen(N) nogroup num_tasks(N) reduction(min:a)
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      for (int j = 0; j < 2; ++j)
        for (int j = 0; j < 2; ++j)
          for (int j = 0; j < 2; ++j)
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      for (int j = 0; j < 2; ++j)
        for (int j = 0; j < 2; ++j)
          for (int j = 0; j < 2; ++j)
            foo();
  // CHECK-NEXT: #pragma omp parallel
  // CHECK-NEXT: #pragma omp taskloop simd private(argc,b) firstprivate(c,d) lastprivate(d,f) collapse(N) shared(g) if(c) final(d) mergeable priority(f) simdlen(N) nogroup num_tasks(N) reduction(min: a)
  // CHECK-NEXT: for (int i = 0; i < 2; ++i)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j)
  // CHECK-NEXT: for (int i = 0; i < 2; ++i)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j)
  // CHECK-NEXT: foo();
  return T();
}

// CHECK-LABEL: int main(int argc, char **argv) {
int main(int argc, char **argv) {
  int b = argc, c, d, e, f, g;
  static int a;
// CHECK: static int a;
#pragma omp taskloop simd if(taskloop: a) default(none) shared(a) final(b) priority(5) safelen(8) linear(b), aligned(argv) nogroup num_tasks(argc) reduction(*: g)
  // CHECK-NEXT: #pragma omp taskloop simd if(taskloop: a) default(none) shared(a) final(b) priority(5) safelen(8) linear(b) aligned(argv) nogroup num_tasks(argc) reduction(*: g)
  for (int i = 0; i < 2; ++i)
    a = 2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp parallel
#pragma omp taskloop simd private(argc, b), firstprivate(argv, c), lastprivate(d, f) collapse(2) shared(g) if(argc) mergeable priority(argc) simdlen(16) grainsize(argc) reduction(max: a, e)
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j)
      foo();
  // CHECK-NEXT: #pragma omp parallel
  // CHECK-NEXT: #pragma omp taskloop simd private(argc,b) firstprivate(argv,c) lastprivate(d,f) collapse(2) shared(g) if(argc) mergeable priority(argc) simdlen(16) grainsize(argc) reduction(max: a,e)
  // CHECK-NEXT: for (int i = 0; i < 10; ++i)
  // CHECK-NEXT: for (int j = 0; j < 10; ++j)
  // CHECK-NEXT: foo();
  return (tmain<int, 5>(argc) + tmain<char, 1>(argv[0][0]));
}

#endif
