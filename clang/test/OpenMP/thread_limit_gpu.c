// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux-gnu -fopenmp-targets=amdgpu-amd-amdhsa -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple amdgpu-amd-amdhsa -fopenmp-targets=amdgpu-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck -check-prefixes=CHECK,CHECK-AMDGPU %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple amdgpu-amd-amdhsa -fopenmp-targets=amdgpu-amd-amdhsa -mllvm -openmp-ir-builder-use-default-max-threads=false -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck -check-prefixes=CHECK,CHECK-AMDGPU-FLAG %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux-gnu -fopenmp-targets=spirv64-intel -emit-llvm-bc %s -o %t-x86-spirv-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple spirv64-intel -fopenmp-targets=spirv64-intel -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-spirv-host.bc -o - | FileCheck -check-prefixes=CHECK,CHECK-SPIRV %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple spirv64-intel -fopenmp-targets=spirv64-intel -mllvm -openmp-ir-builder-use-default-max-threads=false -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-spirv-host.bc -o - | FileCheck -check-prefixes=CHECK,CHECK-SPIRV-FLAG %s
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
#pragma omp target teams distribute parallel for simd ompx_attribute(__attribute__((launch_bounds(42, 84))))
  for (int i = 0; i < N; ++i)
    ;
#pragma omp target teams distribute parallel for simd ompx_attribute(__attribute__((launch_bounds(42, 84)))) num_threads(22)
  for (int i = 0; i < N; ++i)
    ;
#pragma omp target teams distribute parallel for simd ompx_attribute(__attribute__((launch_bounds(42, 84, 86)))) num_threads(20)
  for (int i = 0; i < N; ++i)
    ;
  // A construct split over separate 'target', 'teams' and worksharing
  // directives describes the same kernel as the combined spelling below and
  // must end up with the same thread bounds.
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for num_threads(19)
  for (int i = 0; i < N; ++i)
    ;
#pragma omp target teams distribute parallel for num_threads(19)
  for (int i = 0; i < N; ++i)
    ;
  // thread_limit bounds the size of the contention group, so a num_threads
  // clause asking for more than that cannot raise the bound. Both spellings
  // again have to agree.
#pragma omp target
#pragma omp teams thread_limit(8)
#pragma omp distribute parallel for num_threads(64)
  for (int i = 0; i < N; ++i)
    ;
#pragma omp target teams distribute parallel for thread_limit(8) num_threads(64)
  for (int i = 0; i < N; ++i)
    ;
  // The other way round: a num_threads below the thread_limit is the bound. A
  // 'target' wrapping a combined 'teams distribute parallel for' is scanned for
  // num_threads before the thread_limit clause is known, so the thread_limit
  // must not overwrite the smaller value it already found.
#pragma omp target
#pragma omp teams distribute parallel for num_threads(5) thread_limit(9)
  for (int i = 0; i < N; ++i)
    ;
#pragma omp target teams distribute parallel for num_threads(5) thread_limit(9)
  for (int i = 0; i < N; ++i)
    ;
  // A constant thread_limit still bounds the kernel when num_threads is not a
  // constant, and must not be lost because the non-constant clause was seen
  // first.
#pragma omp target
#pragma omp teams distribute parallel for num_threads(N) thread_limit(7)
  for (int i = 0; i < N; ++i)
    ;
#pragma omp target teams distribute parallel for num_threads(N) thread_limit(7)
  for (int i = 0; i < N; ++i)
    ;
  // And the other way round: a constant num_threads still bounds the kernel
  // when the thread_limit is not a constant.
#pragma omp target
#pragma omp teams distribute parallel for num_threads(6) thread_limit(N)
  for (int i = 0; i < N; ++i)
    ;
#pragma omp target teams distribute parallel for num_threads(6) thread_limit(N)
  for (int i = 0; i < N; ++i)
    ;
}

#endif

// CHECK: define weak_odr protected {{amdgpu|spir}}_kernel void @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+__Z3fooi_}}l14({{.*}}) #[[ATTR1:.+]] {
// CHECK: define weak_odr protected {{amdgpu|spir}}_kernel void @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+__Z3fooi_}}l17({{.*}}) #[[ATTR2:.+]] {
// CHECK: define weak_odr protected {{amdgpu|spir}}_kernel void @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+__Z3fooi_}}l20({{.*}}) #[[ATTR3:.+]] {
// CHECK: define weak_odr protected {{amdgpu|spir}}_kernel void @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+__Z3fooi_}}l23({{.*}}) #[[ATTR4:.+]] {
// CHECK: define weak_odr protected {{amdgpu|spir}}_kernel void @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+__Z3fooi_}}l26({{.*}}) #[[ATTR5:.+]] {
// CHECK: define weak_odr protected {{amdgpu|spir}}_kernel void @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+__Z3fooi_}}l32({{.*}}) #[[SPLIT:.+]] {
// CHECK: define weak_odr protected {{amdgpu|spir}}_kernel void @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+__Z3fooi_}}l37({{.*}}) #[[SPLIT]] {
// CHECK: define weak_odr protected {{amdgpu|spir}}_kernel void @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+__Z3fooi_}}l43({{.*}}) #[[CLAMP:.+]] {
// CHECK: define weak_odr protected {{amdgpu|spir}}_kernel void @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+__Z3fooi_}}l48({{.*}}) #[[CLAMP]] {
// CHECK: define weak_odr protected {{amdgpu|spir}}_kernel void @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+__Z3fooi_}}l55({{.*}}) #[[SMALLER:.+]] {
// CHECK: define weak_odr protected {{amdgpu|spir}}_kernel void @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+__Z3fooi_}}l59({{.*}}) #[[SMALLER]] {
// CHECK: define weak_odr protected {{amdgpu|spir}}_kernel void @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+__Z3fooi_}}l65({{.*}}) #[[DYN_NT:.+]] {
// CHECK: define weak_odr protected {{amdgpu|spir}}_kernel void @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+__Z3fooi_}}l69({{.*}}) #[[DYN_NT]] {
// CHECK: define weak_odr protected {{amdgpu|spir}}_kernel void @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+__Z3fooi_}}l74({{.*}}) #[[DYN_TL:.+]] {
// CHECK: define weak_odr protected {{amdgpu|spir}}_kernel void @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+__Z3fooi_}}l78({{.*}}) #[[DYN_TL]] {

// CHECK-AMDGPU: attributes #[[ATTR1]] = { {{.*}} "amdgpu-flat-work-group-size"="1,256" {{.*}} }
// CHECK-AMDGPU: attributes #[[ATTR2]] = { {{.*}} "amdgpu-flat-work-group-size"="1,4" {{.*}} }
// CHECK-AMDGPU: attributes #[[ATTR3]] = { {{.*}} "amdgpu-flat-work-group-size"="1,42" {{.*}} }
// CHECK-AMDGPU: attributes #[[ATTR4]] = { {{.*}} "amdgpu-flat-work-group-size"="1,22" {{.*}} }
// CHECK-AMDGPU: attributes #[[ATTR5]] = { {{.*}} "amdgpu-flat-work-group-size"="1,20" "amdgpu-max-num-workgroups"="86,1,1" {{.*}} }
// CHECK-AMDGPU: attributes #[[SPLIT]] = { {{.*}} "amdgpu-flat-work-group-size"="1,19" {{.*}} }
// CHECK-AMDGPU: attributes #[[CLAMP]] = { {{.*}} "amdgpu-flat-work-group-size"="1,8" {{.*}} }
// CHECK-AMDGPU: attributes #[[SMALLER]] = { {{.*}} "amdgpu-flat-work-group-size"="1,5" {{.*}} }
// CHECK-AMDGPU: attributes #[[DYN_NT]] = { {{.*}} "amdgpu-flat-work-group-size"="1,7" {{.*}} }
// CHECK-AMDGPU: attributes #[[DYN_TL]] = { {{.*}} "amdgpu-flat-work-group-size"="1,6" {{.*}} }

// CHECK-SPIRV: attributes #[[ATTR1]] = { {{.*}} "omp_target_thread_limit"="256" {{.*}} }
// CHECK-SPIRV: attributes #[[ATTR2]] = { {{.*}} "omp_target_thread_limit"="4"  {{.*}} }
// CHECK-SPIRV: attributes #[[ATTR3]] = { {{.*}} "omp_target_num_teams"="84" "omp_target_thread_limit"="42" {{.*}} }
// CHECK-SPIRV: attributes #[[ATTR4]] = { {{.*}} "omp_target_num_teams"="84" "omp_target_thread_limit"="22" {{.*}} }
// CHECK-SPIRV: attributes #[[ATTR5]] = { {{.*}} "omp_target_num_teams"="84" "omp_target_thread_limit"="20" {{.*}} }
// CHECK-SPIRV: attributes #[[SPLIT]] = { {{.*}} "omp_target_thread_limit"="19" {{.*}} }
// CHECK-SPIRV: attributes #[[CLAMP]] = { {{.*}} "omp_target_thread_limit"="8" {{.*}} }
// CHECK-SPIRV: attributes #[[SMALLER]] = { {{.*}} "omp_target_thread_limit"="5" {{.*}} }
// CHECK-SPIRV: attributes #[[DYN_NT]] = { {{.*}} "omp_target_thread_limit"="7" {{.*}} }
// CHECK-SPIRV: attributes #[[DYN_TL]] = { {{.*}} "omp_target_thread_limit"="6" {{.*}} }

// CHECK-AMDGPU-FLAG: attributes #[[ATTR1]] = {
// CHECK-AMDGPU-FLAG-NOT: amdgpu-flat-work-group-size
// CHECK-AMDGPU-FLAG-NOT: omp_target_thread_limit
// CHECK-AMDGPU-FLAG-SAME: }
// CHECK-AMDGPU-FLAG: attributes #[[ATTR2]] = { {{.*}} "amdgpu-flat-work-group-size"="1,4" {{.*}} }
// CHECK-AMDGPU-FLAG: attributes #[[ATTR3]] = { {{.*}} "amdgpu-flat-work-group-size"="1,42" {{.*}} }
// CHECK-AMDGPU-FLAG: attributes #[[ATTR4]] = { {{.*}} "amdgpu-flat-work-group-size"="1,22" {{.*}} }
// CHECK-AMDGPU-FLAG: attributes #[[ATTR5]] = { {{.*}} "amdgpu-flat-work-group-size"="1,20" "amdgpu-max-num-workgroups"="86,1,1" {{.*}} }
// CHECK-AMDGPU-FLAG: attributes #[[SPLIT]] = { {{.*}} "amdgpu-flat-work-group-size"="1,19" {{.*}} }
// CHECK-AMDGPU-FLAG: attributes #[[CLAMP]] = { {{.*}} "amdgpu-flat-work-group-size"="1,8" {{.*}} }
// CHECK-AMDGPU-FLAG: attributes #[[SMALLER]] = { {{.*}} "amdgpu-flat-work-group-size"="1,5" {{.*}} }
// CHECK-AMDGPU-FLAG: attributes #[[DYN_NT]] = { {{.*}} "amdgpu-flat-work-group-size"="1,7" {{.*}} }
// CHECK-AMDGPU-FLAG: attributes #[[DYN_TL]] = { {{.*}} "amdgpu-flat-work-group-size"="1,6" {{.*}} }

// CHECK-SPIRV-FLAG: attributes #[[ATTR1]] = {
// CHECK-SPIRV-FLAG-NOT: omp_target_thread_limit
// CHECK-SPIRV-FLAG-SAME: }
// CHECK-SPIRV-FLAG: attributes #[[ATTR2]] = { {{.*}} "omp_target_thread_limit"="4"  {{.*}} }
// CHECK-SPIRV-FLAG: attributes #[[ATTR3]] = { {{.*}} "omp_target_num_teams"="84" "omp_target_thread_limit"="42" {{.*}} }
// CHECK-SPIRV-FLAG: attributes #[[ATTR4]] = { {{.*}} "omp_target_num_teams"="84" "omp_target_thread_limit"="22" {{.*}} }
// CHECK-SPIRV-FLAG: attributes #[[ATTR5]] = { {{.*}} "omp_target_num_teams"="84" "omp_target_thread_limit"="20" {{.*}} }
// CHECK-SPIRV-FLAG: attributes #[[SPLIT]] = { {{.*}} "omp_target_thread_limit"="19" {{.*}} }
// CHECK-SPIRV-FLAG: attributes #[[CLAMP]] = { {{.*}} "omp_target_thread_limit"="8" {{.*}} }
// CHECK-SPIRV-FLAG: attributes #[[SMALLER]] = { {{.*}} "omp_target_thread_limit"="5" {{.*}} }
// CHECK-SPIRV-FLAG: attributes #[[DYN_NT]] = { {{.*}} "omp_target_thread_limit"="7" {{.*}} }
// CHECK-SPIRV-FLAG: attributes #[[DYN_TL]] = { {{.*}} "omp_target_thread_limit"="6" {{.*}} }
