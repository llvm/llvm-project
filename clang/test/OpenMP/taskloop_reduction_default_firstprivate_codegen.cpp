// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=60 -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=60 -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

// A taskloop with default(firstprivate) and a reduction, nested inside an
// enclosing captured region (here 'parallel'/'single', with no 'taskgraph'
// involved).  The reduction variable 'res' must be captured *by reference* by
// the enclosing parallel region (its address forwarded to the outlined
// function) so the reduction result propagates back to it.
//
// This is the non-taskgraph form of the same bug fixed for taskgraph: when
// deciding the enclosing region's capture kind, Sema consulted the innermost
// directive's default(firstprivate) instead of the default at the capture
// level, capturing 'res' by copy and severing the reduction write-back (the
// caller then read the unmodified original, i.e. 0).

#ifndef HEADER
#define HEADER

int run(int seed) {
  int x = seed;
  int res = 0;

#pragma omp parallel
#pragma omp single
  {
#pragma omp taskloop num_tasks(4) default(firstprivate) reduction(+ : res)
    for (int i = 0; i < 8; ++i)
      res += x + i;
  }

  return res;
}

#endif

// CHECK-LABEL: define {{.*}}@_Z3runi(
// The reduction variable is captured BY REFERENCE by the enclosing parallel
// region: its address (not its value) is forwarded to the outlined function.
// CHECK:         call {{.*}}@__kmpc_fork_call(ptr {{.*}}, i32 2, ptr @_Z3runi.omp_outlined, ptr %[[RES:[0-9a-z._]+]], ptr %{{[0-9a-z._]+}})
// The caller reads the (reduced) value straight back out of that same storage.
// CHECK:         %[[RET:.*]] = load i32, ptr %[[RES]], align 4
// CHECK-NEXT:    ret i32 %[[RET]]
