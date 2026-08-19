// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -x c++ -triple x86_64-unknown-linux-gnu -fopenmp-targets=x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// Verifies that an 'omp target' region lexically nested inside an
// 'omp taskgraph' is recorded via __kmpc_taskgraph_target, while the same
// region outside a taskgraph still lowers to the ordinary __tgt_target_kernel
// launch. The taskgraph entry point receives the populated
// __tgt_kernel_arguments struct (last pointer argument) together with the
// launch parameters and the depend list.

void outside() {
  int x = 0;
#pragma omp target map(tofrom : x)
  { x++; }
}

void inside() {
  int x = 0;
#pragma omp taskgraph
  {
#pragma omp target map(tofrom : x)
    { x++; }
  }
}

void inside_depend() {
  int x = 0;
#pragma omp taskgraph
  {
#pragma omp target map(tofrom : x) depend(out : x) nowait
    { x++; }
  }
}

// Outside a taskgraph: ordinary kernel launch, no taskgraph recording.
// CHECK-LABEL: define {{.*}}@_Z7outsidev
// CHECK: call i32 @__tgt_target_kernel(
// CHECK-NOT: @__kmpc_taskgraph_target

// Inside a taskgraph: recorded via __kmpc_taskgraph_target. The trailing
// arguments are the __tgt_kernel_arguments struct and the (currently null)
// relocation callback; ndeps (arg 3) is 0 and the dep_list (arg 4) is null for
// a target with no depend clause.
// CHECK: call i32 @__kmpc_taskgraph_target(ptr {{[^,]+}}, i32 {{[^,]+}}, i32 0, ptr null, i32 0, i64 {{[^,]+}}, i32 {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}}, ptr %kernel_args{{[0-9.]*}}, ptr null)

// Inside a taskgraph with depend(out:) + nowait: ndeps is non-zero, the
// dep_list points at the emitted kmp_depend_info array, and has_no_wait is 1.
// CHECK: call i32 @__kmpc_taskgraph_target(ptr {{[^,]+}}, i32 {{[^,]+}}, i32 1, ptr {{[^,]+}}, i32 1, i64 {{[^,]+}}, i32 {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}}, ptr %kernel_args{{[0-9.]*}}, ptr null)

#endif
