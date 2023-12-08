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

template <class T, int N>
T tmain(T argc) {
  T b = argc, c, d, e, f, g;
  static T a;
// CHECK: static T a;
#pragma omp taskgroup allocate(d) task_reduction(+: d)
#pragma omp parallel masked taskloop if(taskloop: argc > N) default(shared) untied priority(N) grainsize(N) reduction(+:g) allocate(g) reduction(task,^:c)
  // CHECK-NEXT: #pragma omp taskgroup allocate(d) task_reduction(+: d)
  // CHECK-NEXT: #pragma omp parallel masked taskloop if(taskloop: argc > N) default(shared) untied priority(N) grainsize(N) reduction(+: g) allocate(g) reduction(task, ^: c){{$}}
  for (int i = 0; i < 2; ++i)
    a = 2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp parallel
#pragma omp parallel masked taskloop private(argc, b), firstprivate(c, d), lastprivate(d, f) collapse(N) shared(g) if (c) final(d) mergeable priority(f) nogroup num_tasks(N)
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      for (int j = 0; j < 2; ++j)
        for (int j = 0; j < 2; ++j)
          for (int j = 0; j < 2; ++j)
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      for (int j = 0; j < 2; ++j)
        for (int j = 0; j < 2; ++j)
          for (int j = 0; j < 2; ++j) {
#pragma omp cancel taskgroup
#pragma omp cancellation point taskgroup
            foo();
          }
  // CHECK-NEXT: #pragma omp parallel
  // CHECK-NEXT: #pragma omp parallel masked taskloop private(argc,b) firstprivate(c,d) lastprivate(d,f) collapse(N) shared(g) if(c) final(d) mergeable priority(f) nogroup num_tasks(N)
  // CHECK-NEXT: for (int i = 0; i < 2; ++i)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j)
  // CHECK-NEXT: for (int i = 0; i < 2; ++i)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j)
  // CHECK-NEXT: for (int j = 0; j < 2; ++j) {
  // CHECK-NEXT: #pragma omp cancel taskgroup
  // CHECK-NEXT: #pragma omp cancellation point taskgroup
  // CHECK-NEXT: foo();
  return T();
}

// CHECK-LABEL: int main(int argc, char **argv) {
int main(int argc, char **argv) {
  int b = argc, c, d, e, f, g;
  int tid = 0;
  static int a;
// CHECK: static int a;
#pragma omp taskgroup task_reduction(+: d)
#pragma omp parallel masked taskloop if(parallel: a) default(none) shared(a, b, argc) final(b) priority(5) num_tasks(argc) reduction(*: g) filter(tid)
  // CHECK-NEXT: #pragma omp taskgroup task_reduction(+: d)
  // CHECK-NEXT: #pragma omp parallel masked taskloop if(parallel: a) default(none) shared(a,b,argc) final(b) priority(5) num_tasks(argc) reduction(*: g) filter(tid)
  for (int i = 0; i < 2; ++i)
    a = 2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp parallel
#pragma omp parallel masked taskloop private(argc, b), firstprivate(argv, c), lastprivate(d, f) collapse(2) shared(g) if(argc) mergeable priority(argc) grainsize(argc) reduction(max: a, e)
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j) {
#pragma omp cancel taskgroup
#pragma omp cancellation point taskgroup
      foo();
    }
  // CHECK-NEXT: #pragma omp parallel
  // CHECK-NEXT: #pragma omp parallel masked taskloop private(argc,b) firstprivate(argv,c) lastprivate(d,f) collapse(2) shared(g) if(argc) mergeable priority(argc) grainsize(argc) reduction(max: a,e)
  // CHECK-NEXT: for (int i = 0; i < 10; ++i)
  // CHECK-NEXT: for (int j = 0; j < 10; ++j) {
  // CHECK-NEXT: #pragma omp cancel taskgroup
  // CHECK-NEXT: #pragma omp cancellation point taskgroup
  // CHECK-NEXT: foo();
  return (tmain<int, 5>(argc) + tmain<char, 1>(argv[0][0]));
}

#endif
