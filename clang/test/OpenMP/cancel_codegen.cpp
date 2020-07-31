// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -triple x86_64-apple-darwin13.4.0 -emit-llvm -o - %s | FileCheck %s --check-prefixes=ALL,CHECK
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple x86_64-apple-darwin13.4.0 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -std=c++11 -include-pch %t -fsyntax-only -verify %s -triple x86_64-apple-darwin13.4.0 -emit-llvm -o - | FileCheck %s --check-prefixes=ALL,CHECK

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-enable-irbuilder -triple x86_64-apple-darwin13.4.0 -emit-llvm -o - %s | FileCheck %s --check-prefixes=ALL,IRBUILDER
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-enable-irbuilder -x c++ -std=c++11 -triple x86_64-apple-darwin13.4.0 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-enable-irbuilder -std=c++11 -include-pch %t -fsyntax-only -verify %s -triple x86_64-apple-darwin13.4.0 -emit-llvm -o - | FileCheck %s --check-prefixes=ALL,IRBUILDER

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -triple x86_64-apple-darwin13.4.0 -emit-llvm -o - %s | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple x86_64-apple-darwin13.4.0 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -std=c++11 -include-pch %t -fsyntax-only -verify %s -triple x86_64-apple-darwin13.4.0 -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

float flag;
int main (int argc, char **argv) {
#pragma omp parallel
{
#pragma omp cancel parallel if(flag)
  argv[0][0] = argc;
#pragma omp barrier
  argv[0][0] += argc;
}
// ALL: call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(
#pragma omp sections
{
#pragma omp cancel sections
}
// ALL: call void @__kmpc_for_static_init_4(
// ALL: call i32 @__kmpc_cancel(
// ALL: call void @__kmpc_for_static_fini(
// ALL: call void @__kmpc_barrier(%struct.ident_t*
#pragma omp sections
{
#pragma omp cancel sections
#pragma omp section
  {
#pragma omp cancel sections
  }
}
// ALL: call void @__kmpc_for_static_init_4(
// ALL: [[RES:%.+]] = call i32 @__kmpc_cancel(%struct.ident_t* {{[^,]+}}, i32 [[GTID:%.*]], i32 3)
// ALL: [[CMP:%.+]] = icmp ne i32 [[RES]], 0
// ALL: br i1 [[CMP]], label %[[EXIT:[^,].+]], label %[[CONTINUE:.+]]
// ALL: [[EXIT]]
// ALL: br label
// ALL: [[CONTINUE]]
// ALL: br label
// ALL: [[RES:%.+]] = call i32 @__kmpc_cancel(%struct.ident_t* {{[^,]+}}, i32 [[GTID:%.*]], i32 3)
// ALL: [[CMP:%.+]] = icmp ne i32 [[RES]], 0
// ALL: br i1 [[CMP]], label %[[EXIT:[^,].+]], label %[[CONTINUE:.+]]
// ALL: [[EXIT]]
// ALL: br label
// ALL: [[CONTINUE]]
// ALL: br label
// ALL: call void @__kmpc_for_static_fini(
#pragma omp for
for (int i = 0; i < argc; ++i) {
#pragma omp cancel for if(cancel: flag)
}
// ALL: call void @__kmpc_for_static_init_4(
// ALL: [[FLAG:%.+]] = load float, float* @{{.+}},
// ALL: [[BOOL:%.+]] = fcmp une float [[FLAG]], 0.000000e+00
// ALL: br i1 [[BOOL]], label %[[THEN:[^,]+]], label %[[ELSE:[^,]+]]
// ALL: [[THEN]]
// ALL: [[RES:%.+]] = call i32 @__kmpc_cancel(%struct.ident_t* {{[^,]+}}, i32 [[GTID:%.*]], i32 2)
// ALL: [[CMP:%.+]] = icmp ne i32 [[RES]], 0
// ALL: br i1 [[CMP]], label %[[EXIT:[^,].+]], label %[[CONTINUE:.+]]
// ALL: [[EXIT]]
// ALL: br label
// ALL: [[CONTINUE]]
// ALL: br label
// ALL: [[ELSE]]
// ALL: br label
// ALL: call void @__kmpc_for_static_fini(
// ALL: call void @__kmpc_barrier(%struct.ident_t*
#pragma omp task
{
#pragma omp cancel taskgroup
}
// ALL: call i8* @__kmpc_omp_task_alloc(
// ALL: call i32 @__kmpc_omp_task(
#pragma omp parallel sections
{
#pragma omp cancel sections
}
// ALL: call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(
#pragma omp parallel sections
{
#pragma omp cancel sections
#pragma omp section
  {
#pragma omp cancel sections
  }
}
// ALL: call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(
int r = 0;
#pragma omp parallel for reduction(+: r)
for (int i = 0; i < argc; ++i) {
#pragma omp cancel for
  r += i;
}
// ALL: call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(
  return argc;
}

// CHECK: define internal void @{{[^(]+}}(i32* {{[^,]+}}, i32* {{[^,]+}},
// CHECK: [[FLAG:%.+]] = load float, float* @{{.+}},
// CHECK: [[BOOL:%.+]] = fcmp une float [[FLAG]], 0.000000e+00
// CHECK: br i1 [[BOOL]], label %[[THEN:[^,]+]], label %[[ELSE:[^,]+]]
// CHECK: [[THEN]]
// CHECK: [[RES:%.+]] = call i32 @__kmpc_cancel(%struct.ident_t* {{[^,]+}}, i32 {{[^,]+}}, i32 1)
// CHECK: [[CMP:%.+]] = icmp ne i32 [[RES]], 0
// CHECK: br i1 [[CMP]], label %[[EXIT:[^,]+]],
// CHECK: [[EXIT]]
// CHECK: br label %[[RETURN:.+]]
// CHECK: [[ELSE]]
// The barrier directive should now call __kmpc_cancel_barrier
// CHECK: call i32 @__kmpc_cancel_barrier(%struct.ident_t*
// CHECK: br label
// CHECK: [[RETURN]]
// CHECK: ret void

// CHECK: define internal i32 @{{[^(]+}}(i32
// CHECK: [[RES:%.+]] = call i32 @__kmpc_cancel(%struct.ident_t* {{[^,]+}}, i32 {{[^,]+}}, i32 4)
// CHECK: [[CMP:%.+]] = icmp ne i32 [[RES]], 0
// CHECK: br i1 [[CMP]], label %[[EXIT:[^,]+]],
// CHECK: [[EXIT]]
// CHECK: br label %[[RETURN:.+]]
// CHECK: [[RETURN]]
// CHECK: ret i32 0

// CHECK: define internal void @{{[^(]+}}(i32* {{[^,]+}}, i32* {{[^,]+}})
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: call i32 @__kmpc_cancel(
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: ret void

// CHECK: define internal void @{{[^(]+}}(i32* {{[^,]+}}, i32* {{[^,]+}})
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: [[RES:%.+]] = call i32 @__kmpc_cancel(%struct.ident_t* {{[^,]+}}, i32 [[GTID:%.+]], i32 3)
// CHECK: [[CMP:%.+]] = icmp ne i32 [[RES]], 0
// CHECK: br i1 [[CMP]], label %[[EXIT:[^,].+]], label %[[CONTINUE:.+]]
// CHECK: [[EXIT]]
// CHECK: br label
// CHECK: [[CONTINUE]]
// CHECK: br label
// CHECK: [[RES:%.+]] = call i32 @__kmpc_cancel(%struct.ident_t* {{[^,]+}}, i32 [[GTID:%.*]], i32 3)
// CHECK: [[CMP:%.+]] = icmp ne i32 [[RES]], 0
// CHECK: br i1 [[CMP]], label %[[EXIT:[^,].+]], label %[[CONTINUE:.+]]
// CHECK: [[EXIT]]
// CHECK: br label
// CHECK: [[CONTINUE]]
// CHECK: br label
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: ret void

// CHECK: define internal void @{{[^(]+}}(i32* {{[^,]+}}, i32* {{[^,]+}},
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: [[RES:%.+]] = call i32 @__kmpc_cancel(%struct.ident_t* {{[^,]+}}, i32 [[GTID:%.+]], i32 2)
// CHECK: [[CMP:%.+]] = icmp ne i32 [[RES]], 0
// CHECK: br i1 [[CMP]], label %[[EXIT:[^,].+]], label %[[CONTINUE:.+]]
// CHECK: [[EXIT]]
// CHECK: br label
// CHECK: [[CONTINUE]]
// CHECK: br label
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: call i32 @__kmpc_reduce_nowait(
// CHECK: call void @__kmpc_end_reduce_nowait(
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: ret void

// IRBUILDER: define internal void @main

// IRBUILDER: [[RETURN:omp.par.outlined.exit[^:]*]]
// IRBUILDER-NEXT: ret void
// IRBUILDER: [[FLAG:%.+]] = load float, float* @{{.+}},

// IRBUILDER: [[BOOL:%.+]] = fcmp une float [[FLAG]], 0.000000e+00
// IRBUILDER: br i1 [[BOOL]], label %[[THEN:[^,]+]], label %[[ELSE:[^,]+]]
// IRBUILDER: [[ELSE]]
// IRBUILDER-NEXT:   br label %[[ELSE2:.*]]
// IRBUILDER: [[ELSE2]]
// The barrier directive should now call __kmpc_cancel_barrier
// IRBUILDER: call i32 @__kmpc_cancel_barrier(%struct.ident_t*
// IRBUILDER: br label
// IRBUILDER: [[THEN]]
// IRBUILDER: [[RES:%.+]] = call i32 @__kmpc_cancel(%struct.ident_t* {{[^,]+}}, i32 {{[^,]+}}, i32 1)
// IRBUILDER: [[CMP:%.+]] = icmp eq i32 [[RES]], 0
// IRBUILDER: br i1 [[CMP]], label %[[CONTINUE:[^,].+]], label %[[EXIT:.+]]
// IRBUILDER: [[EXIT]]
// IRBUILDER: br label %[[RETURN]]
// IRBUILDER: [[CONTINUE]]
// IRBUILDER: br label %[[ELSE2:.+]]

#endif
