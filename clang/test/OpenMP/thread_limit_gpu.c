// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck -check-prefixes=CHECK,CHECK-AMDGPU %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux-gnu -fopenmp-targets=spirv64-intel -emit-llvm-bc %s -o %t-x86-spirv-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple spirv64-intel -fopenmp-targets=spirv64-intel -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-spirv-host.bc -o - | FileCheck -check-prefixes=CHECK,CHECK-SPIRV %s
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

// CHECK: define weak_odr protected {{amdgpu|spir}}_kernel void @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+__Z3fooi_}}l12({{.*}}) #[[ATTR1:.+]] {
// CHECK: define weak_odr protected {{amdgpu|spir}}_kernel void @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+__Z3fooi_}}l15({{.*}}) #[[ATTR2:.+]] {
// CHECK: define weak_odr protected {{amdgpu|spir}}_kernel void @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+__Z3fooi_}}l18({{.*}}) #[[ATTR3:.+]] {
// CHECK: define weak_odr protected {{amdgpu|spir}}_kernel void @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+__Z3fooi_}}l21({{.*}}) #[[ATTR4:.+]] {

// CHECK-AMDGPU: attributes #[[ATTR1]] = { {{.*}} "amdgpu-flat-work-group-size"="1,256" {{.*}} }
// CHECK-AMDGPU: attributes #[[ATTR2]] = { {{.*}} "amdgpu-flat-work-group-size"="1,4" {{.*}} }
// CHECK-AMDGPU: attributes #[[ATTR3]] = { {{.*}} "amdgpu-flat-work-group-size"="1,42" "amdgpu-max-num-workgroups"="42,1,1"{{.*}} }
// CHECK-AMDGPU: attributes #[[ATTR4]] = { {{.*}} "amdgpu-flat-work-group-size"="1,22" "amdgpu-max-num-workgroups"="42,1,1"{{.*}} }

// CHECK-SPIRV: attributes #[[ATTR1]] = { {{.*}} "omp_target_thread_limit"="256" {{.*}} }
// CHECK-SPIRV: attributes #[[ATTR2]] = { {{.*}} "omp_target_thread_limit"="4"  {{.*}} }
// CHECK-SPIRV: attributes #[[ATTR3]] = { {{.*}} "omp_target_num_teams"="42" "omp_target_thread_limit"="42" {{.*}} }
// CHECK-SPIRV: attributes #[[ATTR4]] = { {{.*}} "omp_target_num_teams"="42" "omp_target_thread_limit"="22" {{.*}} }
