// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=60 -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=60 -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

// A taskloop with default(firstprivate) and a reduction, nested inside a
// 'taskgraph' region.  The reduction variable 'res' must be captured *by
// reference* by the enclosing taskgraph region (its address stored into the
// captured record) so the reduction result propagates back to it.  A previous
// bug consulted the innermost directive's default(firstprivate) when deciding
// the enclosing region's capture kind and captured 'res' by copy, severing the
// write-back (the caller then read the unmodified original, i.e. 0).

#ifndef HEADER
#define HEADER

int run(int seed) {
  int x = seed;
  int res = 0;

#pragma omp taskgraph graph_id(1)
  {
#pragma omp taskloop replayable num_tasks(4) default(firstprivate) reduction(+ : res)
    for (int i = 0; i < 8; ++i)
      res += x + i;
  }

  return res;
}

#endif

// CHECK-LABEL: define {{.*}}@_Z3runi(
// CHECK:         %[[CAP:.*]] = alloca %struct.anon, align 8
// The reduction variable is captured BY REFERENCE (a pointer is stored into the
// taskgraph captured record's first field), not by copy.
// CHECK:         %[[FIELD:.*]] = getelementptr inbounds nuw %struct.anon, ptr %[[CAP]], i32 0, i32 0
// CHECK-NEXT:    store ptr %[[RES:.*]], ptr %[[FIELD]], align 8
// CHECK:         call void @__kmpc_taskgraph(
// The caller reads the (reduced) value straight back out of that same storage.
// CHECK:         %[[RET:.*]] = load i32, ptr %[[RES]], align 4
// CHECK-NEXT:    ret i32 %[[RET]]
