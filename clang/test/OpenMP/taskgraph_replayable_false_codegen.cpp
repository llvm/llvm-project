// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -x c++ -triple x86_64-unknown-linux-gnu -fopenmp-targets=x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

// OpenMP 6.0 [14.3]: a task-generating construct encountered in a taskgraph
// construct is a replayable construct of the region "unless otherwise
// specified by the replayable clause".  So the enclosing taskgraph supplies the
// default and the clause overrides it: replayable(false) inside a taskgraph
// region takes the construct out of the record, leaving it to be encountered
// normally on the (recording) execution of the region and, since a replay
// execution does not execute the region body, omitted from every replay.
//
// Checked for each directive that admits the clause: the default records via a
// __kmpc_taskgraph* entry point, replayable(false) takes the ordinary path
// instead, and a non-constant argument -- a clang extension, since [14.6]
// gives the argument the constant property -- selects between the two at run
// time.

#ifndef HEADER
#define HEADER

void body();

// --- task -------------------------------------------------------------------

// CHECK-LABEL: define {{.*}} @_Z{{[0-9]+}}task_defaultv
void task_default() {
  // CHECK: call i32 @__kmpc_taskgraph_task(
  // CHECK-NOT: call i32 @__kmpc_omp_task(
#pragma omp taskgraph
  {
#pragma omp task
    body();
  }
}

// CHECK-LABEL: define {{.*}} @_Z{{[0-9]+}}task_replayable_onev
void task_replayable_one() {
  // An explicit true argument agrees with the default the taskgraph supplies.
  // CHECK: call i32 @__kmpc_taskgraph_task(
  // CHECK-NOT: call i32 @__kmpc_omp_task(
#pragma omp taskgraph
  {
#pragma omp task replayable(1)
    body();
  }
}

// CHECK-LABEL: define {{.*}} @_Z{{[0-9]+}}task_replayable_falsev
void task_replayable_false() {
  // CHECK: call i32 @__kmpc_omp_task(
  // CHECK-NOT: call i32 @__kmpc_taskgraph_task(
#pragma omp taskgraph
  {
#pragma omp task replayable(false)
    body();
  }
}

// CHECK-LABEL: define {{.*}} @_Z{{[0-9]+}}task_replayable_dyni
void task_replayable_dyn(int c) {
  // CHECK: br i1
  // CHECK-DAG: call i32 @__kmpc_taskgraph_task(
  // CHECK-DAG: call i32 @__kmpc_omp_task(
#pragma omp taskgraph
  {
#pragma omp task replayable(c)
    body();
  }
}

// Outside any taskgraph region there is no default to override: a task with no
// clause, and one that opts out, both take the ordinary path.
// CHECK-LABEL: define {{.*}} @_Z{{[0-9]+}}task_outside_defaultv
void task_outside_default() {
  // CHECK: call i32 @__kmpc_omp_task(
  // CHECK-NOT: call i32 @__kmpc_taskgraph_task(
#pragma omp task
  body();
}

// CHECK-LABEL: define {{.*}} @_Z{{[0-9]+}}task_outside_replayable_falsev
void task_outside_replayable_false() {
  // CHECK: call i32 @__kmpc_omp_task(
  // CHECK-NOT: call i32 @__kmpc_taskgraph_task(
#pragma omp task replayable(false)
  body();
}

// --- taskloop ---------------------------------------------------------------

// CHECK-LABEL: define {{.*}} @_Z{{[0-9]+}}taskloop_defaultv
void taskloop_default() {
  // CHECK: call i32 @__kmpc_taskgraph_taskloop(
  // CHECK-NOT: call void @__kmpc_taskloop(
#pragma omp taskgraph
  {
#pragma omp taskloop
    for (int i = 0; i < 4; ++i)
      body();
  }
}

// CHECK-LABEL: define {{.*}} @_Z{{[0-9]+}}taskloop_replayable_falsev
void taskloop_replayable_false() {
  // CHECK: call void @__kmpc_taskloop(
  // CHECK-NOT: call i32 @__kmpc_taskgraph_taskloop(
#pragma omp taskgraph
  {
#pragma omp taskloop replayable(false)
    for (int i = 0; i < 4; ++i)
      body();
  }
}

// CHECK-LABEL: define {{.*}} @_Z{{[0-9]+}}taskloop_replayable_dyni
void taskloop_replayable_dyn(int c) {
  // CHECK: br i1
  // CHECK-DAG: call i32 @__kmpc_taskgraph_taskloop(
  // CHECK-DAG: call void @__kmpc_taskloop(
#pragma omp taskgraph
  {
#pragma omp taskloop replayable(c)
    for (int i = 0; i < 4; ++i)
      body();
  }
}

// --- taskwait ---------------------------------------------------------------

// CHECK-LABEL: define {{.*}} @_Z{{[0-9]+}}taskwait_defaultv
void taskwait_default() {
  int x = 0;
  // CHECK: call void @__kmpc_taskgraph_taskwait(
  // CHECK-NOT: call void @__kmpc_omp_taskwait_deps_51(
#pragma omp taskgraph
  {
#pragma omp taskwait depend(inout : x)
  }
}

// CHECK-LABEL: define {{.*}} @_Z{{[0-9]+}}taskwait_replayable_falsev
void taskwait_replayable_false() {
  int x = 0;
  // CHECK: call void @__kmpc_omp_taskwait_deps_51(
  // CHECK-NOT: call void @__kmpc_taskgraph_taskwait(
#pragma omp taskgraph
  {
#pragma omp taskwait depend(inout : x) replayable(false)
  }
}

// --- target -----------------------------------------------------------------

// CHECK-LABEL: define {{.*}} @_Z{{[0-9]+}}target_defaultv
void target_default() {
  int x = 0;
  // CHECK: call i32 @__kmpc_taskgraph_target(
  // CHECK-NOT: call i32 @__tgt_target_kernel(
#pragma omp taskgraph
  {
#pragma omp target map(tofrom : x)
    { x++; }
  }
}

// CHECK-LABEL: define {{.*}} @_Z{{[0-9]+}}target_replayable_falsev
void target_replayable_false() {
  int x = 0;
  // CHECK: call i32 @__tgt_target_kernel(
  // CHECK-NOT: call i32 @__kmpc_taskgraph_target(
#pragma omp taskgraph
  {
#pragma omp target map(tofrom : x) replayable(false)
    { x++; }
  }
}

// CHECK-LABEL: define {{.*}} @_Z{{[0-9]+}}target_replayable_dyni
void target_replayable_dyn(int c) {
  int x = 0;
  // CHECK: br i1
  // CHECK-DAG: call i32 @__kmpc_taskgraph_target(
  // CHECK-DAG: call i32 @__tgt_target_kernel(
#pragma omp taskgraph
  {
#pragma omp target map(tofrom : x) replayable(c)
    { x++; }
  }
}

// A recorded target forwards its dependences to __kmpc_taskgraph_target, which
// satisfies them synchronously, so it needs no hidden helper task.  One that
// opts out needs the helper task back: it takes the ordinary launch path, where
// the depend clause has nothing else to satisfy it.
// CHECK-LABEL: define {{.*}} @_Z{{[0-9]+}}target_depend_defaultv
void target_depend_default() {
  int x = 0, d = 0;
  // CHECK-NOT: @__kmpc_omp_task_alloc(
  // CHECK: call i32 @__kmpc_taskgraph_target(ptr {{[^,]*}}, i32 {{[^,]*}}, i32 1, ptr
#pragma omp taskgraph
  {
#pragma omp target map(tofrom : x) depend(inout : d)
    { x++; }
  }
}

// CHECK-LABEL: define {{.*}} @_Z{{[0-9]+}}target_depend_replayable_falsev
void target_depend_replayable_false() {
  int x = 0, d = 0;
  // CHECK: call ptr @__kmpc_omp_task_alloc(
  // CHECK: call void @__kmpc_omp_taskwait_deps_51(
  // CHECK-NOT: call i32 @__kmpc_taskgraph_target(
#pragma omp taskgraph
  {
#pragma omp target map(tofrom : x) depend(inout : d) replayable(false)
    { x++; }
  }
}

// --- target enter/exit data, target update ----------------------------------

// CHECK-LABEL: define {{.*}} @_Z{{[0-9]+}}enter_data_defaultv
void enter_data_default() {
  int x = 0;
  // CHECK: call void @__kmpc_taskgraph_target_enter_data(
  // CHECK-NOT: call void @__tgt_target_data_begin_mapper(
#pragma omp taskgraph
  {
#pragma omp target enter data map(to : x)
  }
}

// CHECK-LABEL: define {{.*}} @_Z{{[0-9]+}}enter_data_replayable_falsev
void enter_data_replayable_false() {
  int x = 0;
  // CHECK: call void @__tgt_target_data_begin_mapper(
  // CHECK-NOT: call void @__kmpc_taskgraph_target_enter_data(
#pragma omp taskgraph
  {
#pragma omp target enter data map(to : x) replayable(false)
  }
}

// CHECK-LABEL: define {{.*}} @_Z{{[0-9]+}}exit_data_defaultv
void exit_data_default() {
  int x = 0;
  // CHECK: call void @__kmpc_taskgraph_target_exit_data(
  // CHECK-NOT: call void @__tgt_target_data_end_mapper(
#pragma omp taskgraph
  {
#pragma omp target exit data map(from : x)
  }
}

// CHECK-LABEL: define {{.*}} @_Z{{[0-9]+}}exit_data_replayable_falsev
void exit_data_replayable_false() {
  int x = 0;
  // CHECK: call void @__tgt_target_data_end_mapper(
  // CHECK-NOT: call void @__kmpc_taskgraph_target_exit_data(
#pragma omp taskgraph
  {
#pragma omp target exit data map(from : x) replayable(false)
  }
}

// CHECK-LABEL: define {{.*}} @_Z{{[0-9]+}}update_defaultv
void update_default() {
  int x = 0;
  // CHECK: call void @__kmpc_taskgraph_target_update(
  // CHECK-NOT: call void @__tgt_target_data_update_mapper(
#pragma omp taskgraph
  {
#pragma omp target update to(x)
  }
}

// CHECK-LABEL: define {{.*}} @_Z{{[0-9]+}}update_replayable_falsev
void update_replayable_false() {
  int x = 0;
  // CHECK: call void @__tgt_target_data_update_mapper(
  // CHECK-NOT: call void @__kmpc_taskgraph_target_update(
#pragma omp taskgraph
  {
#pragma omp target update to(x) replayable(false)
  }
}

#endif
