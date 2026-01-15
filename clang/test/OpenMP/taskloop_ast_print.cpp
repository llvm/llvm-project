// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -DOMP60 -ast-print %s | FileCheck %s --check-prefix=CHECK60
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=60 -DOMP60 -ast-print %s | FileCheck %s --check-prefix=CHECK60
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -std=c++11 -include-pch %t -verify %s -ast-print | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

typedef void **omp_impex_t;
extern const omp_impex_t omp_not_impex;
extern const omp_impex_t omp_import;
extern const omp_impex_t omp_export;
extern const omp_impex_t omp_impex;

void foo() {}

template <class T, int N>
T tmain(T argc) {
  T b = argc, c, d, e, f, g;
  static T a;
// CHECK: static T a;
#pragma omp taskgroup allocate(d) task_reduction(+: d)
#pragma omp taskloop if(taskloop: argc > N) default(shared) untied priority(N) grainsize(N) reduction(+:g) in_reduction(+: d) allocate(d)
  // CHECK-NEXT: #pragma omp taskgroup allocate(d) task_reduction(+: d)
  // CHECK-NEXT: #pragma omp taskloop if(taskloop: argc > N) default(shared) untied priority(N) grainsize(N) reduction(+: g) in_reduction(+: d) allocate(d){{$}}
  for (int i = 0; i < 2; ++i)
    a = 2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp parallel
#pragma omp taskloop private(argc, b), firstprivate(c, d), lastprivate(d, f) collapse(N) shared(g) if (c) final(d) mergeable priority(f) nogroup num_tasks(N)
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
  // CHECK-NEXT: #pragma omp taskloop private(argc,b) firstprivate(c,d) lastprivate(d,f) collapse(N) shared(g) if(c) final(d) mergeable priority(f) nogroup num_tasks(N)
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
  static int a;
// CHECK: static int a;
#pragma omp taskgroup task_reduction(+: d)
#pragma omp taskloop if(taskloop: a) default(none) shared(a) final(b) priority(5) num_tasks(argc) reduction(*: g) in_reduction(+:d)
  // CHECK-NEXT: #pragma omp taskgroup task_reduction(+: d)
  // CHECK-NEXT: #pragma omp taskloop if(taskloop: a) default(none) shared(a) final(b) priority(5) num_tasks(argc) reduction(*: g) in_reduction(+: d)
  for (int i = 0; i < 2; ++i)
    a = 2;
// CHECK-NEXT: for (int i = 0; i < 2; ++i)
// CHECK-NEXT: a = 2;
#pragma omp parallel
#pragma omp taskloop private(argc, b), firstprivate(argv, c), lastprivate(d, f) collapse(2) shared(g) if(argc) mergeable priority(argc) grainsize(argc) reduction(max: a, e)
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j) {
#pragma omp cancel taskgroup
#pragma omp cancellation point taskgroup
      foo();
    }
  // CHECK-NEXT: #pragma omp parallel
  // CHECK-NEXT: #pragma omp taskloop private(argc,b) firstprivate(argv,c) lastprivate(d,f) collapse(2) shared(g) if(argc) mergeable priority(argc) grainsize(argc) reduction(max: a,e)
  // CHECK-NEXT: for (int i = 0; i < 10; ++i)
  // CHECK-NEXT: for (int j = 0; j < 10; ++j) {
  // CHECK-NEXT: #pragma omp cancel taskgroup
  // CHECK-NEXT: #pragma omp cancellation point taskgroup
  // CHECK-NEXT: foo();
#ifdef OMP60
#pragma omp taskloop threadset(omp_team)
  for (int i = 0; i < 10; ++i) {
#pragma omp taskloop threadset(omp_pool)
  for (int j = 0; j < 10; ++j) {
    foo();
  }
}

#pragma omp taskloop transparent(omp_not_impex)
  for (int i = 0; i < 10; ++i) {
#pragma omp task transparent(omp_import)
    for (int i = 0; i < 10; ++i) {
#pragma omp task transparent(omp_export)
      for (int i = 0; i < 10; ++i) {
#pragma omp task transparent(omp_impex)
	foo();
      }
    }
  }
#endif
 // CHECK60: #pragma omp taskloop threadset(omp_team)
 // CHECK60-NEXT: for (int i = 0; i < 10; ++i) {
 // CHECK60: #pragma omp taskloop threadset(omp_pool)
 // CHECK60-NEXT: for (int j = 0; j < 10; ++j) {
 // CHECK60-NEXT: foo();

// CHECK60: #pragma omp taskloop transparent(omp_not_impex)
// CHECK60-NEXT: for (int i = 0; i < 10; ++i) {
// CHECK60-NEXT: #pragma omp task transparent(omp_import)
// CHECK60-NEXT: for (int i = 0; i < 10; ++i) {
// CHECK60-NEXT: #pragma omp task transparent(omp_export)
// CHECK60-NEXT: for (int i = 0; i < 10; ++i) {
// CHECK60-NEXT: #pragma omp task transparent(omp_impex)
// CHECK60-NEXT: foo();

  return (tmain<int, 5>(argc) + tmain<char, 1>(argv[0][0]));
}

#endif
