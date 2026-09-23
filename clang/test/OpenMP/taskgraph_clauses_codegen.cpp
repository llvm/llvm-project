// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -x c++ -std=c++11 -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=60 -x c++ -std=c++11 -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=60 -x c++ -std=c++11 -triple x86_64-unknown-unknown -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

// Codegen for the graph_id and graph_reset clauses.  The arguments of interest
// are the 4th (graph_id, i64) and 5th (graph_reset, i32) of __kmpc_taskgraph.

#ifndef HEADER
#define HEADER

void body();

// CHECK-LABEL: define {{.*}} @_Z10no_clausesv
void no_clauses() {
  // CHECK: call void @__kmpc_taskgraph(ptr {{.*}}, i32 %{{.*}}, ptr @.omp.taskgraph.handle{{[^,]*}}, i64 0, i32 0, i32 0,
#pragma omp taskgraph
  { body(); }
}

// An omitted graph_reset argument has the optional property: it means "reset",
// so it must lower to a constant 1 -- not to 0, and not to a null-pointer
// dereference in the frontend.
// CHECK-LABEL: define {{.*}} @_Z17reset_no_argumentv
void reset_no_argument() {
  // CHECK: call void @__kmpc_taskgraph(ptr {{.*}}, i32 %{{.*}}, ptr @.omp.taskgraph.handle{{[^,]*}}, i64 0, i32 1, i32 0,
#pragma omp taskgraph graph_reset
  { body(); }
}

// A true condition must agree with the omitted form rather than differing by a
// sign extension (it used to pass -1).
// CHECK-LABEL: define {{.*}} @_Z10reset_truev
void reset_true() {
  // CHECK: call void @__kmpc_taskgraph(ptr {{.*}}, i32 %{{.*}}, ptr @.omp.taskgraph.handle{{[^,]*}}, i64 0, i32 1, i32 0,
#pragma omp taskgraph graph_reset(true)
  { body(); }
}

// CHECK-LABEL: define {{.*}} @_Z11reset_falsev
void reset_false() {
  // CHECK: call void @__kmpc_taskgraph(ptr {{.*}}, i32 %{{.*}}, ptr @.omp.taskgraph.handle{{[^,]*}}, i64 0, i32 0, i32 0,
#pragma omp taskgraph graph_reset(false)
  { body(); }
}

// CHECK-LABEL: define {{.*}} @_Z10reset_exprb
void reset_expr(bool c) {
  // CHECK: [[TOBOOL:%.*]] = icmp ne i8 %{{.*}}, 0
  // CHECK-NEXT: [[RESET:%.*]] = zext i1 [[TOBOOL]] to i32
  // CHECK-NEXT: call void @__kmpc_taskgraph(ptr {{.*}}, i32 %{{.*}}, ptr @.omp.taskgraph.handle{{[^,]*}}, i64 0, i32 [[RESET]], i32 0,
#pragma omp taskgraph graph_reset(c)
  { body(); }
}

// An omitted argument combines with graph_id as usual.
// CHECK-LABEL: define {{.*}} @_Z24id_and_reset_no_argumenti
void id_and_reset_no_argument(int id) {
  // CHECK: [[GID:%.*]] = zext i32 %{{.*}} to i64
  // CHECK-NEXT: call void @__kmpc_taskgraph(ptr {{.*}}, i32 %{{.*}}, ptr @.omp.taskgraph.handle{{[^,]*}}, i64 [[GID]], i32 1, i32 0,
#pragma omp taskgraph graph_id(id) graph_reset
  { body(); }
}

#endif
