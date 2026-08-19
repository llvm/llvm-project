// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -x c -triple x86_64-unknown-linux-gnu -fopenmp-targets=x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// A firstprivate aggregate of a target construct in a taskgraph region carries
// the SAVED map bit when its value belongs to the saved data environment, which
// makes the recording snapshot its bytes.  That is the case when the clause
// says 'saved', and when the variable is declared inside the taskgraph region:
// a replay does not run its declaration again (OpenMP 6.0 [14.3]).  A
// firstprivate of a variable in the taskgraph's enclosing data environment is
// not saved.

struct S {
  int A[16];
  int B;
};

void outer(void) {
  struct S s;
#pragma omp taskgraph
  {
#pragma omp target firstprivate(s)
    { s.B = 1; }
  }
}

void saved(void) {
  struct S s;
#pragma omp taskgraph
  {
#pragma omp target firstprivate(saved : s)
    { s.B = 1; }
  }
}

void inner(int n) {
#pragma omp taskgraph
  {
    struct S s;
    s.B = n;
#pragma omp target firstprivate(s)
    { s.B += 1; }
  }
}

// outer: PRIVATE | TO | TARGET_PARAM.
// CHECK: @.offload_maptypes{{[.0-9]*}} = private unnamed_addr constant [{{[0-9]+}} x i64] [i64 161{{[],]}}
// saved, and inner without being asked: the same, and SAVED.
// CHECK: @.offload_maptypes{{[.0-9]*}} = private unnamed_addr constant [{{[0-9]+}} x i64] [i64 65697{{[],]}}
// CHECK: @.offload_maptypes{{[.0-9]*}} = private unnamed_addr constant [{{[0-9]+}} x i64] [i64 65697{{[],]}}

#endif
