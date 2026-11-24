// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 -std=c++11 \
// RUN:   -fopenmp-metadirective-runtime-selection -ast-print %s \
// RUN:   | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 -std=c++11 \
// RUN:   -fopenmp-metadirective-runtime-selection -x c++ -emit-pch -o %t %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 -std=c++11 \
// RUN:   -fopenmp-metadirective-runtime-selection -include-pch %t \
// RUN:   -ast-print %s | FileCheck %s

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// CHECK-LABEL: void when_otherwise(int flag) {
// CHECK-NEXT: #pragma omp metadirective when(user={condition(flag)}: parallel) otherwise(single)
// CHECK-NEXT: {
void when_otherwise(int flag) {
#pragma omp metadirective when(user={condition(flag)}: parallel) otherwise(single)
  {
  }
}

// The implicit 'otherwise' is not printed.
// CHECK-LABEL: void no_otherwise(int flag) {
// CHECK-NEXT: #pragma omp metadirective when(user={condition(flag)}: parallel){{$}}
void no_otherwise(int flag) {
#pragma omp metadirective when(user={condition(flag)}: parallel)
  {
  }
}

// Variant clauses and an empty variant are printed as written.
// CHECK-LABEL: void clauses(int flag) {
// CHECK-NEXT: #pragma omp metadirective when(user={condition(flag)}: parallel num_threads(4)) when(user={condition(!flag)}: )
void clauses(int flag) {
#pragma omp metadirective when(user={condition(flag)}: parallel num_threads(4)) when(user={condition(!flag)}: )
  {
  }
}

// A loop variant prints its loop, not just the loop body.
// CHECK-LABEL: void loop(int flag, int n) {
// CHECK-NEXT: #pragma omp metadirective when(user={condition(flag)}: parallel for) otherwise(for)
// CHECK-NEXT: for (int i = 0; i < n; ++i)
void loop(int flag, int n) {
#pragma omp metadirective when(user={condition(flag)}: parallel for) otherwise(for)
  for (int i = 0; i < n; ++i)
    ;
}

// A standalone variant is printed in its clause; the statement once.
// CHECK-LABEL: void standalone(int flag, int n) {
// CHECK-NEXT: #pragma omp metadirective when(user={condition(flag)}: barrier) otherwise(single)
// CHECK-NEXT: n = 1;
// CHECK-NEXT: }
void standalone(int flag, int n) {
#pragma omp metadirective when(user={condition(flag)}: barrier) otherwise(single)
  n = 1;
}

// All variants are standalone: the statement follows the metadirective.
// CHECK-LABEL: void all_standalone(int flag, int n) {
// CHECK-NEXT: #pragma omp metadirective when(user={condition(flag)}: barrier) otherwise(taskyield)
// CHECK-NEXT: n = 1;
// CHECK-NEXT: }
void all_standalone(int flag, int n) {
#pragma omp metadirective when(user={condition(flag)}: barrier) otherwise(taskyield)
  n = 1;
}

// 'target update' is standalone: the statement is printed once.
// CHECK-LABEL: void target_update(int flag, int *p) {
// CHECK-NEXT: #pragma omp metadirective when(user={condition(flag)}: target update to(p[0:1])) otherwise(parallel)
// CHECK-NEXT: *p = 1;
// CHECK-NEXT: }
void target_update(int flag, int *p) {
#pragma omp metadirective when(user={condition(flag)}: target update to(p[0:1])) otherwise(parallel)
  *p = 1;
}

// 'cancel' and 'cancellation point' print their construct-type clause.
// CHECK-LABEL: void cancel_region(int flag) {
// CHECK: #pragma omp metadirective when(user={condition(flag)}: cancel parallel) otherwise(cancellation point parallel)
void cancel_region(int flag) {
#pragma omp parallel
  {
#pragma omp metadirective when(user={condition(flag)}: cancel parallel) otherwise(cancellation point parallel)
  }
}

void bar();

// A named 'critical' prints its name.
// CHECK-LABEL: void named_critical(int flag) {
// CHECK-NEXT: #pragma omp metadirective when(user={condition(flag)}: critical (lck) hint(0)) otherwise(single)
// CHECK-NEXT: bar();
void named_critical(int flag) {
#pragma omp metadirective when(user={condition(flag)}: critical(lck) hint(0)) otherwise(single)
  bar();
}

#endif
