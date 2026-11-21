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

#endif
