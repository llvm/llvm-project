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

  // 'saved' on an omp task lexically inside a taskgraph.
  // CHECK: #pragma omp taskgraph
  // CHECK: #pragma omp task firstprivate(saved: a)
  #pragma omp taskgraph
  {
    #pragma omp task firstprivate(saved: a)
    { (void)a; }
  }

  // Multiple variables.
  // CHECK: #pragma omp taskgraph
  // CHECK: #pragma omp task firstprivate(saved: a,b,c)
  #pragma omp taskgraph
  {
    #pragma omp task firstprivate(saved: a, b, c)
    { (void)a; (void)b; (void)c; }
  }

  // Mixed with another clause.
  // CHECK: #pragma omp taskgraph
  // CHECK: #pragma omp task firstprivate(saved: a) shared(b)
  #pragma omp taskgraph
  {
    #pragma omp task firstprivate(saved: a) shared(b)
    { (void)a; (void)b; }
  }

  // 'saved' on an omp taskloop lexically inside a taskgraph.
  // CHECK: #pragma omp taskgraph
  // CHECK: #pragma omp taskloop firstprivate(saved: a)
  #pragma omp taskgraph
  {
    #pragma omp taskloop firstprivate(saved: a)
    for (int i = 0; i < 4; ++i) (void)a;
  }

  // 'saved' on a replayable omp task outside any taskgraph - also legal.
  // CHECK: #pragma omp task replayable firstprivate(saved: a)
  #pragma omp task replayable firstprivate(saved: a)
  { (void)a; }

  // 'saved' on a replayable omp taskloop outside any taskgraph - also legal.
  // CHECK: #pragma omp taskloop replayable firstprivate(saved: a)
  #pragma omp taskloop replayable firstprivate(saved: a)
  for (int i = 0; i < 4; ++i) (void)a;

  // 'saved' on a non-lexically-nested task (dynamic nesting via a call into
  // a function from a taskgraph region is the runtime use case) - we accept
  // any task/taskloop construct since the static check cannot prove dynamic
  // nesting.
  // CHECK: #pragma omp task firstprivate(saved: a)
  #pragma omp task firstprivate(saved: a)
  { (void)a; }
}

#endif
