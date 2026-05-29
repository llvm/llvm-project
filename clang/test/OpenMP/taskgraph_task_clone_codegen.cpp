// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// Verifies that for a 'firstprivate' on a task inside a taskgraph whose
// type has a non-trivial copy constructor, the compiler emits a dedicated
// '.omp_task_clone.' helper and passes it to __kmpc_taskgraph_task in
// the trailing argument slot.  The helper re-runs the copy constructor
// from the origin task's '.kmp_privates.t' field into the clone's, so
// that the runtime memcpy does not produce a torn copy of a non-
// trivially-copyable object.

struct NonTrivial {
  int v;
  int *self;
  NonTrivial();
  NonTrivial(const NonTrivial &other);
  ~NonTrivial();
};

void run() {
  NonTrivial nt;
#pragma omp taskgraph
  {
#pragma omp task firstprivate(nt)
    {
      (void)nt.v;
    }
  }
}

// The clone helper is passed as the trailing pointer argument to
// __kmpc_taskgraph_task (10 total: ident, gtid, task, flags, sizes...,
// ndeps, deps, relocation, clone).
// CHECK: call i32 @__kmpc_taskgraph_task(ptr {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}}, i32 {{[^,]+}}, i64 {{[^,]+}}, i64 {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}}, ptr {{[^,]+}}, ptr @.omp_task_clone.)

// The clone helper has the same calling convention as the existing
// taskloop task-dup callback so that the runtime can dispatch through
// a single function-pointer type; the third parameter is unused here.
// The body indexes the same .kmp_privates.t field on both source and
// destination tasks and invokes NonTrivial's copy constructor.
// CHECK: define internal void @.omp_task_clone.(ptr noundef %{{[^,]+}}, ptr noundef %{{[^,]+}}, i32 noundef %{{[^,]+}})
// CHECK: getelementptr inbounds {{.*}} %struct.kmp_task_t_with_privates,
// CHECK: getelementptr inbounds {{.*}} %struct.kmp_task_t_with_privates,
// CHECK: getelementptr inbounds {{.*}} %struct..kmp_privates.t,
// CHECK: getelementptr inbounds {{.*}} %struct..kmp_privates.t,
// CHECK: call void @_ZN10NonTrivialC1ERKS_(

#endif
