// RUN: %clang_cc1 -verify -fopenmp -triple x86_64-apple-darwin13.4.0 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-apple-darwin13.4.0 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -fsyntax-only -verify %s -triple x86_64-apple-darwin13.4.0 -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics
// REQUIRES: x86-registered-target
#ifndef HEADER
#define HEADER

int main (int argc, char **argv) {
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(
#pragma omp parallel
{
#pragma omp cancellation point parallel
#pragma omp cancel parallel
  argv[0][0] = argc;
}
// CHECK: call void (%ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(
#pragma omp sections
{
  {
#pragma omp cancellation point sections
#pragma omp cancel sections
  }
}
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: [[RES:%.+]] = call i32 @__kmpc_cancellationpoint(%ident_t* {{[^,]+}}, i32 [[GTID]], i32 3)
// CHECK: [[CMP:%.+]] = icmp ne i32 [[RES]], 0
// CHECK: br i1 [[CMP]], label %[[EXIT:[^,].+]], label %[[CONTINUE:.+]]
// CHECK: [[EXIT]]
// CHECK: call i32 @__kmpc_cancel_barrier(%ident_t*
// CHECK: br label
// CHECK: [[CONTINUE]]
// CHECK: br label
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: call void @__kmpc_barrier(%ident_t*
#pragma omp sections
{
#pragma omp cancellation point sections
#pragma omp section
  {
#pragma omp cancellation point sections
#pragma omp cancel sections
  }
}
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: [[RES:%.+]] = call i32 @__kmpc_cancellationpoint(%ident_t* {{[^,]+}}, i32 [[GTID]], i32 3)
// CHECK: [[CMP:%.+]] = icmp ne i32 [[RES]], 0
// CHECK: br i1 [[CMP]], label %[[EXIT:[^,].+]], label %[[CONTINUE:.+]]
// CHECK: [[EXIT]]
// CHECK: call i32 @__kmpc_cancel_barrier(%ident_t*
// CHECK: br label
// CHECK: [[CONTINUE]]
// CHECK: br label
// CHECK: [[RES:%.+]] = call i32 @__kmpc_cancellationpoint(%ident_t* {{[^,]+}}, i32 [[GTID]], i32 3)
// CHECK: [[CMP:%.+]] = icmp ne i32 [[RES]], 0
// CHECK: br i1 [[CMP]], label %[[EXIT:[^,].+]], label %[[CONTINUE:.+]]
// CHECK: [[EXIT]]
// CHECK: call i32 @__kmpc_cancel_barrier(%ident_t*
// CHECK: br label
// CHECK: [[CONTINUE]]
// CHECK: br label
// CHECK: call void @__kmpc_for_static_fini(
#pragma omp for
for (int i = 0; i < argc; ++i) {
#pragma omp cancellation point for
#pragma omp cancel for
}
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: [[RES:%.+]] = call i32 @__kmpc_cancellationpoint(%ident_t* {{[^,]+}}, i32 [[GTID]], i32 2)
// CHECK: [[CMP:%.+]] = icmp ne i32 [[RES]], 0
// CHECK: br i1 [[CMP]], label %[[EXIT:[^,].+]], label %[[CONTINUE:.+]]
// CHECK: [[EXIT]]
// CHECK: call i32 @__kmpc_cancel_barrier(%ident_t*
// CHECK: br label
// CHECK: [[CONTINUE]]
// CHECK: br label
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: call void @__kmpc_barrier(%ident_t*
#pragma omp task
{
#pragma omp cancellation point taskgroup
#pragma omp cancel taskgroup
}
// CHECK: call i8* @__kmpc_omp_task_alloc(
// CHECK: call i32 @__kmpc_omp_task(
#pragma omp parallel sections
{
  {
#pragma omp cancellation point sections
#pragma omp cancel sections
  }
}
// CHECK: call void (%ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(
#pragma omp parallel sections
{
  {
#pragma omp cancellation point sections
#pragma omp cancel sections
  }
#pragma omp section
  {
#pragma omp cancellation point sections
  }
}
// CHECK: call void (%ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(
#pragma omp parallel for
for (int i = 0; i < argc; ++i) {
#pragma omp cancellation point for
#pragma omp cancel for
}
// CHECK: call void (%ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(
  return argc;
}

// CHECK: define internal void @{{[^(]+}}(i32* {{[^,]+}}, i32* {{[^,]+}},
// CHECK: [[RES:%.+]] = call i32 @__kmpc_cancellationpoint(%ident_t* {{[^,]+}}, i32 {{[^,]+}}, i32 1)
// CHECK: [[CMP:%.+]] = icmp ne i32 [[RES]], 0
// CHECK: br i1 [[CMP]], label %[[EXIT:[^,]+]],
// CHECK: [[EXIT]]
// CHECK: call i32 @__kmpc_cancel_barrier(%ident_t*
// CHECK: br label %[[RETURN:.+]]
// CHECK: [[RETURN]]
// CHECK: ret void

// CHECK: define internal i32 @{{[^(]+}}(i32
// CHECK: [[RES:%.+]] = call i32 @__kmpc_cancellationpoint(%ident_t* {{[^,]+}}, i32 {{[^,]+}}, i32 4)
// CHECK: [[CMP:%.+]] = icmp ne i32 [[RES]], 0
// CHECK: br i1 [[CMP]], label %[[EXIT:[^,]+]],
// CHECK: [[EXIT]]
// CHECK: call i32 @__kmpc_cancel_barrier(%ident_t*
// CHECK: br label %[[RETURN:.+]]
// CHECK: [[RETURN]]
// CHECK: ret i32 0

// CHECK: define internal void @{{[^(]+}}(i32* {{[^,]+}}, i32* {{[^,]+}})
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: [[RES:%.+]] = call i32 @__kmpc_cancellationpoint(%ident_t* {{[^,]+}}, i32 [[GTID:%.+]], i32 3)
// CHECK: [[CMP:%.+]] = icmp ne i32 [[RES]], 0
// CHECK: br i1 [[CMP]], label %[[EXIT:[^,].+]], label %[[CONTINUE:.+]]
// CHECK: [[EXIT]]
// CHECK: call i32 @__kmpc_cancel_barrier(%ident_t*
// CHECK: br label
// CHECK: [[CONTINUE]]
// CHECK: br label
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: ret void

// CHECK: define internal void @{{[^(]+}}(i32* {{[^,]+}}, i32* {{[^,]+}})
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: [[RES:%.+]] = call i32 @__kmpc_cancellationpoint(%ident_t* {{[^,]+}}, i32 [[GTID:%.+]], i32 3)
// CHECK: [[CMP:%.+]] = icmp ne i32 [[RES]], 0
// CHECK: br i1 [[CMP]], label %[[EXIT:[^,].+]], label %[[CONTINUE:.+]]
// CHECK: [[EXIT]]
// CHECK: call i32 @__kmpc_cancel_barrier(%ident_t*
// CHECK: br label
// CHECK: [[CONTINUE]]
// CHECK: br label
// CHECK: [[RES:%.+]] = call i32 @__kmpc_cancellationpoint(%ident_t* {{[^,]+}}, i32 [[GTID]], i32 3)
// CHECK: [[CMP:%.+]] = icmp ne i32 [[RES]], 0
// CHECK: br i1 [[CMP]], label %[[EXIT:[^,].+]], label %[[CONTINUE:.+]]
// CHECK: [[EXIT]]
// CHECK: call i32 @__kmpc_cancel_barrier(%ident_t*
// CHECK: br label
// CHECK: [[CONTINUE]]
// CHECK: br label
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: ret void

// CHECK: define internal void @{{[^(]+}}(i32* {{[^,]+}}, i32* {{[^,]+}},
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: [[RES:%.+]] = call i32 @__kmpc_cancellationpoint(%ident_t* {{[^,]+}}, i32 [[GTID:%.+]], i32 2)
// CHECK: [[CMP:%.+]] = icmp ne i32 [[RES]], 0
// CHECK: br i1 [[CMP]], label %[[EXIT:[^,].+]], label %[[CONTINUE:.+]]
// CHECK: [[EXIT]]
// CHECK: call i32 @__kmpc_cancel_barrier(%ident_t*
// CHECK: br label
// CHECK: [[CONTINUE]]
// CHECK: br label
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: ret void

#endif
