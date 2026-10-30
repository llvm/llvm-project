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

#endif
