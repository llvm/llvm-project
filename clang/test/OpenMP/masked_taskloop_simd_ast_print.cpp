// RUN: %clang_cc1 -verify -fopenmp -DOMP5 -ast-print %s | FileCheck %s --check-prefix CHECK --check-prefix OMP50
// RUN: %clang_cc1 -fopenmp -DOMP5 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -DOMP5 -std=c++11 -include-pch %t -verify %s -ast-print | FileCheck %s --check-prefix CHECK --check-prefix OMP50
// RUN: %clang_cc1 -verify -fopenmp -ast-print %s -DOMP5 | FileCheck %s --check-prefix CHECK --check-prefix OMP50
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s -DOMP5
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -verify %s -ast-print -DOMP5 | FileCheck %s --check-prefix CHECK --check-prefix OMP50

// RUN: %clang_cc1 -verify -fopenmp-simd -DOMP5 -ast-print %s | FileCheck %s --check-prefix CHECK --check-prefix OMP50
// RUN: %clang_cc1 -fopenmp-simd -DOMP5 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -DOMP5 -std=c++11 -include-pch %t -verify %s -ast-print | FileCheck %s --check-prefix CHECK --check-prefix OMP50
// RUN: %clang_cc1 -verify -fopenmp-simd -ast-print %s -DOMP5 | FileCheck %s --check-prefix CHECK --check-prefix OMP50
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -emit-pch -o %t %s -DOMP5
// RUN: %clang_cc1 -fopenmp-simd -std=c++11 -include-pch %t -verify %s -ast-print -DOMP5 | FileCheck %s --check-prefix CHECK --check-prefix OMP50
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
#pragma omp taskgroup task_reduction(+: d) allocate(d)
#pragma omp masked taskloop simd allocate(d) if(taskloop: argc > N) default(shared) untied priority(N) safelen(N) linear(c) aligned(ptr) grainsize(N) reduction(+:g) in_reduction(+: d)
  // CHECK-NEXT: #pragma omp taskgroup task_reduction(+: d) allocate(d)
  // CHECK-NEXT: #pragma omp masked taskloop simd allocate(d) if(taskloop: argc > N) default(shared) untied priority(N) safelen(N) linear(c) aligned(ptr) grainsize(N) reduction(+: g) in_reduction(+: d){{$}}
  for (int i = 0; i < 2; ++i)
    a = 2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp parallel
#pragma omp masked taskloop simd private(argc, b), firstprivate(c, d), lastprivate(d, f) collapse(N) shared(g) if (c) final(d) mergeable priority(f) simdlen(N) nogroup num_tasks(N)
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
  // CHECK-NEXT: #pragma omp masked taskloop simd private(argc,b) firstprivate(c,d) lastprivate(d,f) collapse(N) shared(g) if(c) final(d) mergeable priority(f) simdlen(N) nogroup num_tasks(N)
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
  int b = argc, c, d, e, f, g, h;
  int tid = 0;
  static int a;
// CHECK: static int a;
#pragma omp taskgroup task_reduction(+: d)
#pragma omp masked taskloop simd if(taskloop: a) default(none) shared(a) final(b) priority(5) safelen(8) linear(b), aligned(argv) num_tasks(argc) reduction(*: g) in_reduction(+: d) filter(tid)
  // CHECK-NEXT: #pragma omp taskgroup task_reduction(+: d)
  // CHECK-NEXT: #pragma omp masked taskloop simd if(taskloop: a) default(none) shared(a) final(b) priority(5) safelen(8) linear(b) aligned(argv) num_tasks(argc) reduction(*: g) in_reduction(+: d) filter(tid)
  for (int i = 0; i < 2; ++i)
    a = 2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp parallel
#ifdef OMP5
#pragma omp masked taskloop simd private(argc, b), firstprivate(argv, c), lastprivate(d, f) collapse(2) shared(g) if(simd:argc) mergeable priority(argc) simdlen(16) grainsize(argc) reduction(max: a, e) nontemporal(argc, c, d) order(concurrent) filter(tid)
#else
#pragma omp masked taskloop simd private(argc, b), firstprivate(argv, c), lastprivate(d, f) collapse(2) shared(g) if(argc) mergeable priority(argc) simdlen(16) grainsize(argc) reduction(max: a, e)
#endif // OMP5
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j)
      foo();
  // CHECK-NEXT: #pragma omp parallel
  // OMP50-NEXT: #pragma omp masked taskloop simd private(argc,b) firstprivate(argv,c) lastprivate(d,f) collapse(2) shared(g) if(simd: argc) mergeable priority(argc) simdlen(16) grainsize(argc) reduction(max: a,e) nontemporal(argc,c,d) order(concurrent) filter(tid)
  // CHECK-NEXT: for (int i = 0; i < 10; ++i)
  // CHECK-NEXT: for (int j = 0; j < 10; ++j)
  // CHECK-NEXT: foo();
  return (tmain<int, 5>(argc) + tmain<char, 1>(argv[0][0]));
}

#endif
