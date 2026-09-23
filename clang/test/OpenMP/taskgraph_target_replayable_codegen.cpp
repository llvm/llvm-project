// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -x c++ -triple x86_64-unknown-linux-gnu -fopenmp-targets=x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// Verifies that a 'replayable' clause routes a target / target-data construct
// to the __kmpc_taskgraph_target* entry points even when the construct is not
// lexically nested inside an 'omp taskgraph', mirroring how 'omp task' handles
// the clause:
//   * replayable with no argument (constant-true): emitIfClause folds away the
//     ordinary path, leaving a single taskgraph recording call.
//   * replayable(expr): emitIfClause emits a runtime branch -- the taskgraph
//     recording call when the condition holds, the ordinary launch otherwise.

void replayable_const() {
  int x = 0;
#pragma omp target map(tofrom : x) replayable
  { x++; }
}

void replayable_cond(int c) {
  int x = 0;
#pragma omp target map(tofrom : x) replayable(c)
  { x++; }
}

void replayable_enter_data_const() {
  int x = 0;
#pragma omp target enter data map(to : x) replayable
}

void replayable_update_cond(int c) {
  int x = 0;
#pragma omp target update to(x) replayable(c)
}

// A replayable target with a depend clause must forward the dependences to
// __kmpc_taskgraph_target itself (the entry point satisfies them
// synchronously) and must NOT be wrapped in a hidden helper task: the recording
// path executes synchronously.
void replayable_depend_const() {
  int x = 0, d = 0;
#pragma omp target map(tofrom : x) depend(inout : d) replayable
  { x++; }
}

// A replayable target whose region is a teams-distribute construct: the
// recording path evaluates the launch bounds (number of teams, thread limit and
// the loop trip count) just as the ordinary path does, and those expressions
// refer to `n`, which is captured by the target region.  So the recording path
// has to emit them inside an inlined region with CapturedStmtInfo installed;
// emitting them in the enclosing function's context instead used to assert in
// CodeGenFunction::EmitDeclRefLValue.
void replayable_teams_distribute(double *p, int n) {
#pragma omp target replayable
  {
#pragma omp teams distribute
    for (int i = 0; i < n; ++i)
      p[i] = 0.0;
  }
}

// 'replayable' has none of the innermost-leaf, outermost-leaf, all-privatizing
// or once-for-all-constituents clause properties, so it takes the default
// all-constituents property and applies to every leaf construct of a compound
// directive that accepts it.  For a compound directive in the target family
// that is the 'target' leaf alone, so the clause is accepted here and routes
// the construct to __kmpc_taskgraph_target exactly as it does on a lone
// 'target'.
void replayable_combined(double *p, int n) {
#pragma omp target teams replayable map(tofrom : p[0 : n])
  { p[0] = 1.0; }
#pragma omp target teams distribute parallel for replayable map(tofrom : p[0 : n])
  for (int i = 0; i < n; ++i)
    p[i] = 0.0;
}

// Constant-true replayable target: only the taskgraph recording call is
// emitted; the ordinary __tgt_target_kernel launch is folded away.
// CHECK-LABEL: define {{.*}}@_Z16replayable_constv
// CHECK: call i32 @__kmpc_taskgraph_target(
// CHECK-NOT: @__tgt_target_kernel

// replayable(c): a runtime branch selects between the taskgraph recording call
// and the ordinary launch.
// CHECK-LABEL: define {{.*}}@_Z15replayable_condi
// CHECK: br i1
// CHECK-DAG: call i32 @__kmpc_taskgraph_target(
// CHECK-DAG: call i32 @__tgt_target_kernel(

// Constant-true replayable 'target enter data': only the taskgraph recording
// call is emitted.
// CHECK-LABEL: define {{.*}}@_Z27replayable_enter_data_constv
// CHECK: call void @__kmpc_taskgraph_target_enter_data(
// CHECK-NOT: @__tgt_target_data_begin_mapper

// replayable(c) 'target update': a runtime branch selects between the taskgraph
// recording call and the ordinary mapper call.
// CHECK-LABEL: define {{.*}}@_Z22replayable_update_condi
// CHECK: br i1
// CHECK-DAG: call void @__kmpc_taskgraph_target_update(
// CHECK-DAG: call void @__tgt_target_data_update_mapper(

// Constant-true replayable target with a depend clause: the dependences are
// passed to __kmpc_taskgraph_target (non-zero ndeps + a depend list) and no
// hidden helper task is created.
// CHECK-LABEL: define {{.*}}@_Z23replayable_depend_constv
// CHECK-NOT: @__kmpc_omp_task_alloc
// CHECK: call i32 @__kmpc_taskgraph_target(ptr {{[^,]*}}, i32 {{[^,]*}}, i32 1, ptr
// CHECK-NOT: @__kmpc_omp_task_alloc

// Replayable target around a teams-distribute region: still just the taskgraph
// recording call, with the launch bounds computed for it.
// CHECK-LABEL: define {{.*}}@_Z27replayable_teams_distributePdi
// CHECK: call i32 @__kmpc_taskgraph_target(
// CHECK-NOT: @__tgt_target_kernel

// 'replayable' on a compound directive of the target family applies to the
// 'target' leaf, so both constructs record rather than launch.
// CHECK-LABEL: define {{.*}}@_Z19replayable_combinedPdi
// CHECK: call i32 @__kmpc_taskgraph_target(
// CHECK: call i32 @__kmpc_taskgraph_target(
// CHECK-NOT: @__tgt_target_kernel

#endif
