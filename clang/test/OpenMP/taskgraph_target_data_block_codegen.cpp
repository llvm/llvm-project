// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -x c++ -triple x86_64-unknown-linux-gnu -fopenmp-targets=x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// Verifies that the 'omp target data' block construct is recorded into a
// taskgraph as three nodes: an enter-data action, a task whose region is the
// body, and an exit-data action.  The body task carries the taskgraph-body
// flag (0x400, so 1025 with tiedness), runs its region inside a taskgroup, and
// converts its use-device list items itself when it starts.  Outside a
// taskgraph, and when replayable(false) opts out, the construct still lowers
// to the ordinary __tgt_target_data_{begin,end}_mapper pair with the body
// inline.
//
// The standalone 'target {enter|exit} data' / 'target update' directives are
// covered by taskgraph_target_data_codegen.cpp.

void outside() {
  int x = 0;
#pragma omp target data map(tofrom : x)
  { x += 1; }
}

void inside() {
  int x = 0;
#pragma omp taskgraph
  {
#pragma omp target data map(tofrom : x)
    { x += 1; }
  }
}

// The use-device list item is answered by the body task when it starts, so
// the recorded actions do not ask for it: RETURN_PARAM (0x40) is dropped from
// their map types, leaving 0x3.
void usedev(int *p, int n) {
#pragma omp taskgraph
  {
#pragma omp target data map(tofrom : p[0 : n]) use_device_ptr(p)
    { p[0] = 0; }
  }
}

// A 'replayable' clause records the construct outside a taskgraph too.
void replayable_clause() {
  int x = 0;
#pragma omp target data map(tofrom : x) replayable
  { x += 1; }
}

// A 'present' modifier is dropped from the exit action's map types, exactly as
// it is from the ordinary end-mapper call, so the two actions pass different
// map-type arrays.
void present() {
  int x = 0;
#pragma omp taskgraph
  {
#pragma omp target data map(present, tofrom : x)
    { x += 1; }
  }
}

// An 'if' clause suppresses the mapping while leaving the body alone, so it
// guards each action and not the body.
void ifclause(int c) {
  int x = 0;
#pragma omp taskgraph
  {
#pragma omp target data map(tofrom : x) if (c)
    { x += 1; }
  }
}

void optout() {
  int x = 0;
#pragma omp taskgraph
  {
#pragma omp target data map(tofrom : x) replayable(0)
    { x += 1; }
  }
}

// The map types of the 'present' case: 0x1003 (present|to|from) for the enter
// action, and 0x3 for the exit action, which drops 'present'.
// CHECK: @[[MT_USEDEV:[.0-9A-Za-z_]+]] = private unnamed_addr constant [2 x i64] [i64 3, i64 16384]
// CHECK: @[[MT_PRESENT:[.0-9A-Za-z_]+]] = private unnamed_addr constant [1 x i64] [i64 4099]
// CHECK-NEXT: @[[MT_END:[.0-9A-Za-z_]+]] = private unnamed_addr constant [1 x i64] [i64 3]

// Outside a taskgraph: the ordinary mapper pair, no taskgraph recording.
// CHECK-LABEL: define {{.*}}@_Z7outsidev
// CHECK: call void @__tgt_target_data_begin_mapper(
// CHECK: call void @__tgt_target_data_end_mapper(
// CHECK-NOT: call void @__kmpc_taskgraph_target_

// Inside a taskgraph: an enter-data node, the body task, then an exit-data
// node.  With no depend clause, ndeps (arg 3) is 0 and dep_list (arg 4) is
// null; the seven map-info pointers are forwarded, followed by the relocation
// callback.  Both actions share the base-pointer and pointer arrays, which are
// filled in once ahead of the enter action so that they dominate both calls --
// and so share one callback too, there being one pair of arrays to refresh.
// The body task has a relocation helper of its own, for its shareds.
// CHECK-LABEL: define {{.*}}@_Z6insidev
// CHECK-LABEL: define internal void @taskgraph.omp_outlined.(
// CHECK: call void @__kmpc_taskgraph_target_enter_data(ptr {{[^,]+}}, i32 {{[^,]+}}, i32 0, ptr null, i32 0, i64 {{[^,]+}}, i32 1, ptr [[BP:%[^,]+]], ptr [[P:%[^,]+]], ptr {{[^,]+}}, ptr {{[^,]+}}, ptr null, ptr null, ptr [[RELOC:@[.0-9A-Za-z_]+]])
// CHECK: call ptr @__kmpc_omp_task_alloc(ptr {{[^,]+}}, i32 {{[^,]+}}, i32 1025, i64 {{[0-9]+}}, i64 {{[0-9]+}}, ptr [[BODY:@[.0-9A-Za-z_]+]])
// CHECK: call i32 @__kmpc_taskgraph_task(ptr {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}}, i32 1025, i64 {{[0-9]+}}, i64 {{[0-9]+}}, i32 0, ptr null, ptr @.omp.taskgraph.relocate.{{[.0-9]*}}, ptr null)
// CHECK: call void @__kmpc_taskgraph_target_exit_data(ptr {{[^,]+}}, i32 {{[^,]+}}, i32 0, ptr null, i32 0, i64 {{[^,]+}}, i32 1, ptr [[BP]], ptr [[P]], ptr {{[^,]+}}, ptr {{[^,]+}}, ptr null, ptr null, ptr [[RELOC]])
// CHECK-NOT: call void @__tgt_target_data_begin_mapper(
// CHECK: define internal {{.*}}i32 [[BODY]](
// CHECK-NOT: __tgt_taskgraph_lookup_use_device
// CHECK: call void @__kmpc_taskgroup(
// CHECK: add nsw i32
// CHECK: call void @__kmpc_end_taskgroup(

// CHECK-LABEL: define {{.*}}@_Z6usedevPii
// CHECK-LABEL: define internal void @taskgraph.omp_outlined..{{[0-9]+}}(
// CHECK: call void @__kmpc_taskgraph_target_enter_data({{.*}}, ptr @[[MT_USEDEV]], ptr null, ptr null, ptr @{{[.0-9A-Za-z_]+}})
// CHECK: call ptr @__kmpc_omp_task_alloc(ptr {{[^,]+}}, i32 {{[^,]+}}, i32 1025, i64 {{[0-9]+}}, i64 {{[0-9]+}}, ptr [[UDBODY:@[.0-9A-Za-z_]+]])
// CHECK: call void @__kmpc_taskgraph_target_exit_data({{.*}}, ptr @[[MT_USEDEV]], ptr null, ptr null, ptr @{{[.0-9A-Za-z_]+}})
// CHECK: define internal {{.*}}i32 [[UDBODY]](
// CHECK: store i64 67, ptr
// CHECK: call void @__tgt_taskgraph_lookup_use_device(i64 -1, i32 1, ptr {{[^,]+}}, ptr {{[^,]+}}, ptr {{[^,]+}}, ptr [[OUT:%[^,)]+]])
// CHECK: call void @__kmpc_taskgroup(
// CHECK: load ptr, ptr [[OUT]]
// CHECK: call void @__kmpc_end_taskgroup(

// CHECK-LABEL: define {{.*}}@_Z17replayable_clausev
// CHECK: call void @__kmpc_taskgraph_target_enter_data(
// CHECK: call i32 @__kmpc_taskgraph_task(ptr {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}}, i32 1025,
// CHECK: call void @__kmpc_taskgraph_target_exit_data(

// CHECK-LABEL: define {{.*}}@_Z7presentv
// CHECK: call void @__kmpc_taskgraph_target_enter_data({{.*}}, ptr @[[MT_PRESENT]], ptr null, ptr null, ptr @{{[.0-9A-Za-z_]+}})
// CHECK: call void @__kmpc_taskgraph_target_exit_data({{.*}}, ptr @[[MT_END]], ptr null, ptr null, ptr @{{[.0-9A-Za-z_]+}})

// Each action sits in its own guarded block, and both branch on the same
// already-evaluated condition rather than re-evaluating the clause.
// CHECK-LABEL: define {{.*}}@_Z8ifclausei
// CHECK: [[COND:%[a-z0-9.]+]] = icmp ne i32 {{%[0-9]+}}, 0
// CHECK: br i1 [[COND]], label %[[THEN1:[^,]+]], label %[[END1:[a-z0-9._]+]]
// CHECK: [[THEN1]]:
// CHECK: call void @__kmpc_taskgraph_target_enter_data(
// CHECK: [[END1]]:
// CHECK: br i1 [[COND]], label %[[THEN2:[^,]+]], label %{{[a-z0-9._]+}}
// CHECK: [[THEN2]]:
// CHECK: call void @__kmpc_taskgraph_target_exit_data(

// replayable(false): the ordinary mapper pair, even inside the taskgraph.
// CHECK-LABEL: define {{.*}}@_Z6optoutv
// CHECK: call void @__tgt_target_data_begin_mapper(
// CHECK: call void @__tgt_target_data_end_mapper(
// CHECK-NOT: call void @__kmpc_taskgraph_target_

#endif
