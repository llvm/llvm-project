// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

void foo(int N) {
#pragma omp target teams distribute parallel for simd
  for (int i = 0; i < N; ++i)
    ;
#pragma omp target teams distribute parallel for simd thread_limit(4)
  for (int i = 0; i < N; ++i)
    ;
#pragma omp target teams distribute parallel for simd ompx_attribute(__attribute__((launch_bounds(42, 42))))
  for (int i = 0; i < N; ++i)
    ;
#pragma omp target teams distribute parallel for simd ompx_attribute(__attribute__((launch_bounds(42, 42)))) num_threads(22)
  for (int i = 0; i < N; ++i)
    ;
}

#endif

// CHECK: define weak_odr protected amdgpu_kernel void @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+__Z3fooi_}}l10({{.*}}) #[[ATTR1:.+]] {
// CHECK: define weak_odr protected amdgpu_kernel void @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+__Z3fooi_}}l13({{.*}}) #[[ATTR2:.+]] {
// CHECK: define weak_odr protected amdgpu_kernel void @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+__Z3fooi_}}l16({{.*}}) #[[ATTR3:.+]] {
// CHECK: define weak_odr protected amdgpu_kernel void @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+__Z3fooi_}}l19({{.*}}) #[[ATTR4:.+]] {

// CHECK: attributes #[[ATTR1]] = { {{.*}} "amdgpu-flat-work-group-size"="1,256" {{.*}} }
// CHECK: attributes #[[ATTR2]] = { {{.*}} "amdgpu-flat-work-group-size"="1,4" {{.*}} }
// CHECK: attributes #[[ATTR3]] = { {{.*}} "amdgpu-flat-work-group-size"="1,42" "amdgpu-max-num-workgroups"="42,1,1"{{.*}} }
// CHECK: attributes #[[ATTR4]] = { {{.*}} "amdgpu-flat-work-group-size"="1,22" "amdgpu-max-num-workgroups"="42,1,1"{{.*}} }
