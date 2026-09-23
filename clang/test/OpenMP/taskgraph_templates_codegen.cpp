// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// Exercises taskgraph codegen with orthogonal language/runtime features:
//  - C++ templates + non-trivial firstprivate cloning.
//  - taskgroup task_reduction / task in_reduction inside taskgraph.
//  - taskwait depend(...) inside taskgraph (task-generating path).

template <typename T>
struct Box {
  T v;
  Box();
  Box(const Box &);
  ~Box();
};

template <typename T>
void templated_clone(T seed) {
  Box<T> B;
  B.v = seed;
#pragma omp taskgraph
  {
#pragma omp task firstprivate(B)
    {
      (void)B.v;
    }

#pragma omp task replayable(false) firstprivate(B)
    {
      (void)B.v;
    }
  }
}

template <typename T>
T templated_task_reduction(T seed) {
  T Acc = seed;
#pragma omp taskgraph
  {
#pragma omp taskgroup task_reduction(+: Acc)
    {
#pragma omp task in_reduction(+: Acc)
      {
        Acc += seed;
      }
    }
#pragma omp taskwait depend(in: Acc)
  }
  return Acc;
}

int main() {
  templated_clone<int>(7);
  return templated_task_reduction<int>(1);
}

// The taskgraph task entry uses the taskgraph runtime path and carries a
// clone helper for non-trivial firstprivate copies.
// CHECK: call i32 @__kmpc_taskgraph_task(
// CHECK-SAME: ptr @.omp_task_clone.)
// CHECK: define internal void @.omp_task_clone.(

// task_reduction in a taskgraph uses the dedicated taskgraph reduction init.
// CHECK: call ptr @__kmpc_taskgraph_taskred_init(

// taskwait depend(...) inside taskgraph uses the dedicated taskgraph taskwait
// entry point instead of the generic taskwait runtime path.
// CHECK: call void @__kmpc_taskgraph_taskwait(

#endif
