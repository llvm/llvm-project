// Check no warnings/errors
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fsyntax-only -verify %s
// expected-no-diagnostics

// Check AST and unparsing
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -ast-dump  %s | FileCheck %s --check-prefix=DUMP

// Check same results after serialization round-trip
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -emit-pch -o %t %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -include-pch %t -ast-dump-all %s | FileCheck %s --check-prefix=DUMP

#ifndef HEADER
#define HEADER

#define N 12
int A[N];
int B[N];


// DUMP-LABEL:  FunctionDecl {{.*}} main
int
main() {

  for (int i = 0; i < N; ++i) {
    A[i] = 0;
  }

  // assume is for the "simd" region
  // DUMP:      OMPSimdDirective
  // DUMP-NEXT:   CapturedStmt
  // DUMP-NEXT:   CapturedDecl
  #pragma omp assume no_openmp
  #pragma omp simd
  for (int i = 0; i < N; ++i){
    A[i] += B[i];
  }
  // DUMP:  OMPAssumeAttr {{.*}} "omp_no_openmp"

  return 0;
}

#endif
