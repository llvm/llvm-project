// Check no warnings/errors
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -fsyntax-only -verify %s
// expected-no-diagnostics

// Check unparsing
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -ast-print %s | FileCheck %s

// Check same results after serialization round-trip
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -emit-pch -o %t %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -include-pch %t -ast-print %s | FileCheck %s

#ifndef HEADER
#define HEADER

void firstprivate_saved() {
  int a = 1;
  int b = 2;
  int c = 3;

  // CHECK: #pragma omp task firstprivate(a)
  #pragma omp task firstprivate(a)
  { (void)a; }

  // CHECK: #pragma omp task firstprivate(saved: a)
  #pragma omp task firstprivate(saved: a)
  { (void)a; }

  // CHECK: #pragma omp task firstprivate(saved: a,b,c)
  #pragma omp task firstprivate(saved: a, b, c)
  { (void)a; (void)b; (void)c; }

  // CHECK: #pragma omp task firstprivate(saved: a) shared(b)
  #pragma omp task firstprivate(saved: a) shared(b)
  { (void)a; (void)b; }

  // CHECK: #pragma omp taskloop firstprivate(saved: a)
  #pragma omp taskloop firstprivate(saved: a)
  for (int i = 0; i < 4; ++i) (void)a;
}

#endif
