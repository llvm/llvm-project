// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -verify -fopenmp -fexceptions -fcxx-exceptions -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

int a;

// CHECK-NOT: @__omp_offloading_{{.+}}_exec_mode = weak constant i8 1
// CHECK-DAG: [[DISTR_FULL:@.+]] = private unnamed_addr constant %struct.ident_t { i32 0, i32 2050, i32 0, i32 {{[0-9]+}}, i8* getelementptr inbounds
// CHECK-DAG: [[FULL:@.+]] = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 {{[0-9]+}}, i8* getelementptr inbounds
// CHECK-DAG: [[BAR_FULL:@.+]] = private unnamed_addr constant %struct.ident_t { i32 0, i32 66, i32 0, i32 {{[0-9]+}}, i8* getelementptr inbounds
// CHECK-NOT: @__omp_offloading_{{.+}}_exec_mode = weak constant i8 1

void foo() {
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[DISTR_FULL]]
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[DISTR_FULL]]
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[DISTR_FULL]]
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[DISTR_FULL]]
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[DISTR_FULL]]
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[DISTR_FULL]]
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[DISTR_FULL]]
// CHECK-DAG: [[FULL]]
#pragma omp target teams distribute parallel for simd if(a)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target teams distribute parallel for simd schedule(static)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target teams distribute parallel for simd schedule(static, 1)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target teams distribute parallel for simd schedule(auto)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target teams distribute parallel for simd schedule(runtime)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target teams distribute parallel for simd schedule(dynamic)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target teams distribute parallel for simd schedule(guided)
  for (int i = 0; i < 10; ++i)
    ;
int a;
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[DISTR_FULL]]
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[DISTR_FULL]]
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[DISTR_FULL]]
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[DISTR_FULL]]
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[DISTR_FULL]]
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[DISTR_FULL]]
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[DISTR_FULL]]
// CHECK-DAG: [[FULL]]
#pragma omp target teams distribute parallel for lastprivate(a)
  for (int i = 0; i < 10; ++i)
    a = i;
#pragma omp target teams distribute parallel for schedule(static)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target teams distribute parallel for schedule(static, 1)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target teams distribute parallel for schedule(auto)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target teams distribute parallel for schedule(runtime)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target teams distribute parallel for schedule(dynamic)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target teams distribute parallel for schedule(guided)
  for (int i = 0; i < 10; ++i)
    ;
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[DISTR_FULL]]
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[DISTR_FULL]]
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init(
// CHECK-DAG: [[DISTR_FULL]]
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[DISTR_FULL]]
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[DISTR_FULL]]
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[DISTR_FULL]]
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[DISTR_FULL]]
// CHECK-DAG: [[FULL]]
#pragma omp target teams
   {
     int b;
#pragma omp distribute parallel for simd
  for (int i = 0; i < 10; ++i)
    ;
  ;
   }
#pragma omp target teams
   {
     int b[] = {2, 3, sizeof(int)};
#pragma omp distribute parallel for simd schedule(static)
  for (int i = 0; i < 10; ++i)
    ;
   }
#pragma omp target teams
   {
     int b;
#pragma omp distribute parallel for simd schedule(static, 1)
  for (int i = 0; i < 10; ++i)
    ;
  int &c = b;
   }
#pragma omp target teams
#pragma omp distribute parallel for simd schedule(auto)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target teams
#pragma omp distribute parallel for simd schedule(runtime)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target teams
#pragma omp distribute parallel for simd schedule(dynamic)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target teams
#pragma omp distribute parallel for simd schedule(guided)
  for (int i = 0; i < 10; ++i)
    ;
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[DISTR_FULL]]
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[DISTR_FULL]]
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[DISTR_FULL]]
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[DISTR_FULL]]
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[DISTR_FULL]]
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[DISTR_FULL]]
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[DISTR_FULL]]
// CHECK-DAG: [[FULL]]
#pragma omp target teams
#pragma omp distribute parallel for
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target teams
#pragma omp distribute parallel for schedule(static)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target teams
#pragma omp distribute parallel for schedule(static, 1)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target teams
#pragma omp distribute parallel for schedule(auto)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target teams
#pragma omp distribute parallel for schedule(runtime)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target teams
#pragma omp distribute parallel for schedule(dynamic)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target teams
#pragma omp distribute parallel for schedule(guided)
  for (int i = 0; i < 10; ++i)
    ;
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[DISTR_FULL]]
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[DISTR_FULL]]
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[DISTR_FULL]]
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[DISTR_FULL]]
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[DISTR_FULL]]
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[DISTR_FULL]]
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[FULL]]
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for schedule(static)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for schedule(static, 1)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for schedule(auto)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for schedule(runtime)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for schedule(dynamic)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for schedule(guided)
  for (int i = 0; i < 10; ++i)
    ;
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[FULL]]
#pragma omp target parallel for if(a)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target parallel for schedule(static)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target parallel for schedule(static, 1)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target parallel for schedule(auto)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target parallel for schedule(runtime)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target parallel for schedule(dynamic)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target parallel for schedule(guided)
  for (int i = 0; i < 10; ++i)
    ;
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[FULL]]
// CHECK-DAG: [[BAR_FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[FULL]]
// CHECK-DAG: [[BAR_FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[FULL]]
// CHECK-DAG: [[BAR_FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[FULL]]
// CHECK-DAG: [[BAR_FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[FULL]]
// CHECK-DAG: [[BAR_FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[FULL]]
// CHECK-DAG: [[BAR_FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[FULL]]
// CHECK-DAG: [[BAR_FULL]]
#pragma omp target parallel if(a)
#pragma omp for simd
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target parallel
#pragma omp for simd schedule(static)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target parallel
#pragma omp for simd schedule(static, 1)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target parallel
#pragma omp for simd schedule(auto)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target parallel
#pragma omp for simd schedule(runtime)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target parallel
#pragma omp for simd schedule(dynamic)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target parallel
#pragma omp for simd schedule(guided)
  for (int i = 0; i < 10; ++i)
    ;
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[FULL]]
// CHECK-DAG: [[BAR_FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[FULL]]
// CHECK-DAG: [[BAR_FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[FULL]]
// CHECK-DAG: [[BAR_FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[FULL]]
// CHECK-DAG: [[BAR_FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[FULL]]
// CHECK-DAG: [[BAR_FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[FULL]]
// CHECK-DAG: [[BAR_FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[FULL]]
#pragma omp target
#pragma omp parallel
#pragma omp for simd ordered
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp parallel
#pragma omp for simd schedule(static)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp parallel
#pragma omp for simd schedule(static, 1)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp parallel
#pragma omp for simd schedule(auto)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp parallel
#pragma omp for simd schedule(runtime)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp parallel
#pragma omp for simd schedule(dynamic)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp parallel
#pragma omp for simd schedule(guided)
  for (int i = 0; i < 10; ++i)
    ;
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[FULL]]
// CHECK: call i32 @__kmpc_target_init({{.*}}, i8 2, i1 false)
// CHECK-DAG: [[FULL]]
#pragma omp target
#pragma omp parallel for
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp parallel for schedule(static)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp parallel for schedule(static, 1)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp parallel for schedule(auto)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp parallel for schedule(runtime)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target
#pragma omp parallel for schedule(guided)
  for (int i = 0; i < 10; ++i)
    ;
}

#endif

