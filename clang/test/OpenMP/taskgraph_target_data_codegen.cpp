// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -x c++ -triple x86_64-unknown-linux-gnu -fopenmp-targets=x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// Verifies that 'omp target enter data', 'omp target exit data', and
// 'omp target update' directives lexically nested inside an 'omp taskgraph'
// are recorded via the matching __kmpc_taskgraph_target_{enter,exit}_data /
// _update entry points (with the map-info arrays forwarded), while the same
// directives outside a taskgraph still lower to the ordinary
// __tgt_target_data_*_mapper calls.

void outside() {
  int x = 0;
#pragma omp target enter data map(to : x)
#pragma omp target exit data map(from : x)
}

void inside() {
  int x = 0;
#pragma omp taskgraph
  {
#pragma omp target enter data map(to : x)
#pragma omp target update to(x)
#pragma omp target exit data map(from : x)
  }
}

// Outside a taskgraph: ordinary data mapper calls, no taskgraph recording.
// CHECK-LABEL: define {{.*}}@_Z7outsidev
// CHECK: call void @__tgt_target_data_begin_mapper(
// CHECK: call void @__tgt_target_data_end_mapper(
// CHECK-NOT: @__kmpc_taskgraph_target_

// Inside a taskgraph: recorded via the matching taskgraph entry points. With
// no depend clause, ndeps (arg 3) is 0 and dep_list (arg 4) is null; the seven
// map-info pointers are forwarded, followed by the (currently null) relocation
// callback as the trailing argument.
// CHECK: call void @__kmpc_taskgraph_target_enter_data(ptr {{[^,]+}}, i32 {{[^,]+}}, i32 0, ptr null, i32 0, i64 {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}}, ptr {{[^,]+}}, ptr {{[^,]+}}, ptr {{[^,]+}}, ptr {{[^,]+}}, ptr {{[^,]+}}, ptr null)
// CHECK: call void @__kmpc_taskgraph_target_update(ptr {{[^,]+}}, i32 {{[^,]+}}, i32 0, ptr null, i32 0, i64 {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}}, ptr {{[^,]+}}, ptr {{[^,]+}}, ptr {{[^,]+}}, ptr {{[^,]+}}, ptr {{[^,]+}}, ptr null)
// CHECK: call void @__kmpc_taskgraph_target_exit_data(ptr {{[^,]+}}, i32 {{[^,]+}}, i32 0, ptr null, i32 0, i64 {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}}, ptr {{[^,]+}}, ptr {{[^,]+}}, ptr {{[^,]+}}, ptr {{[^,]+}}, ptr {{[^,]+}}, ptr null)

#endif
