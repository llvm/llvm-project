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

// Per OpenMP 6.0 [14.3], a 'firstprivate' clause with the 'saved' modifier on
// a replayable construct may include variables with static storage duration;
// they are copied into the saved data environment of the taskgraph record.
// This covers file-scope statics, static-local variables, static data
// members, and const-qualified statics, all of which Sema accepts and Clang
// codegen places into the per-task '.kmp_privates.t' tail struct.

static int FileScopeStatic = 100;
static const int FileScopeConstStatic = 200;

struct WithStaticMember {
  static int StaticMember;
  static const int StaticConstMember = 400;
};
int WithStaticMember::StaticMember = 0;

void firstprivate_saved_statics() {
  static int LocalStatic = 300;
  static const int LocalConstStatic = 500;

  // CHECK-LABEL: void firstprivate_saved_statics
  // CHECK: #pragma omp task firstprivate(saved: FileScopeStatic)
  #pragma omp task firstprivate(saved: FileScopeStatic)
  { (void)FileScopeStatic; }

  // CHECK: #pragma omp task firstprivate(saved: FileScopeConstStatic)
  #pragma omp task firstprivate(saved: FileScopeConstStatic)
  { (void)FileScopeConstStatic; }

  // CHECK: #pragma omp task firstprivate(saved: LocalStatic)
  #pragma omp task firstprivate(saved: LocalStatic)
  { (void)LocalStatic; }

  // CHECK: #pragma omp task firstprivate(saved: LocalConstStatic)
  #pragma omp task firstprivate(saved: LocalConstStatic)
  { (void)LocalConstStatic; }

  // CHECK: #pragma omp task firstprivate(saved: WithStaticMember::StaticMember)
  #pragma omp task firstprivate(saved: WithStaticMember::StaticMember)
  { (void)WithStaticMember::StaticMember; }

  // CHECK: #pragma omp task firstprivate(saved: WithStaticMember::StaticConstMember)
  #pragma omp task firstprivate(saved: WithStaticMember::StaticConstMember)
  { (void)WithStaticMember::StaticConstMember; }

  // Multiple statics in a single clause, mixed with a non-static.
  int local_int = 0;
  // CHECK: #pragma omp task firstprivate(saved: FileScopeStatic,LocalStatic,WithStaticMember::StaticMember,local_int)
  #pragma omp task firstprivate(saved:                                         \
                                FileScopeStatic, LocalStatic,                  \
                                WithStaticMember::StaticMember, local_int)
  {
    (void)FileScopeStatic;
    (void)LocalStatic;
    (void)WithStaticMember::StaticMember;
    (void)local_int;
  }

  // Same on a 'taskloop' construct.
  // CHECK: #pragma omp taskloop firstprivate(saved: FileScopeStatic,LocalConstStatic)
  #pragma omp taskloop firstprivate(saved: FileScopeStatic, LocalConstStatic)
  for (int i = 0; i < 4; ++i) {
    (void)FileScopeStatic;
    (void)LocalConstStatic;
  }
}

#endif
