// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -fopenmp-cuda-force-full-runtime | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - -fopenmp-cuda-force-full-runtime | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -fexceptions -fcxx-exceptions -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - -fopenmp-cuda-force-full-runtime | FileCheck %s
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK-NOT: @__omp_offloading_{{.+}}_exec_mode = weak constant i8 1

void foo() {
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
#pragma omp target teams distribute parallel for simd
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
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
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
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
#pragma omp target teams
#pragma omp distribute parallel for simd
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target teams
#pragma omp distribute parallel for simd schedule(static)
  for (int i = 0; i < 10; ++i)
    ;
#pragma omp target teams
#pragma omp distribute parallel for simd schedule(static, 1)
  for (int i = 0; i < 10; ++i)
    ;
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
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
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
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
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
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
#pragma omp target parallel for
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
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
#pragma omp target parallel
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
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
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
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
// CHECK: call i32 @__kmpc_target_init({{.*}}, i1 true, i1 false, i1 true)
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

