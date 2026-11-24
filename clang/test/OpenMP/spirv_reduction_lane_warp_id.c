// The inter-warp copy of a GPU reduction gets the lane and warp ids from the
// runtime warp size on SPIR-V, whose subgroup size is not known at compile
// time, and from GV_Warp_Size elsewhere.

// RUN: %clang_cc1 -verify -fopenmp -x c -triple x86_64-unknown-linux -fopenmp-targets=spirv64-intel -emit-llvm-bc %s -o %t-intel-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c -triple spirv64-intel -fopenmp-targets=spirv64-intel -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-intel-host.bc -o - | FileCheck %s --check-prefix=SPIRV

// RUN: %clang_cc1 -verify -fopenmp -x c -triple x86_64-unknown-linux -fopenmp-targets=spirv64-amd-amdhsa -emit-llvm-bc %s -o %t-amd-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c -triple spirv64-amd-amdhsa -fopenmp-targets=spirv64-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-amd-host.bc -o - | FileCheck %s --check-prefix=SPIRV

// RUN: %clang_cc1 -verify -fopenmp -x c -triple x86_64-unknown-linux -fopenmp-targets=spirv64 -emit-llvm-bc %s -o %t-spirv-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c -triple spirv64 -fopenmp-targets=spirv64 -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-spirv-host.bc -o - | FileCheck %s --check-prefix=SPIRV

// RUN: %clang_cc1 -verify -fopenmp -x c -triple x86_64-unknown-linux -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-nvptx-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c -triple nvptx64-nvidia-cuda -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-nvptx-host.bc -o - | FileCheck %s --check-prefix=NVPTX

// expected-no-diagnostics

// SPIRV-LABEL: define internal {{.*}}void @_omp_reduction_inter_warp_copy_func(
// SPIRV:         call {{.*}}i32 @__kmpc_get_hardware_thread_id_in_block()
// SPIRV-NEXT:    [[TID0:%.*]] = call {{.*}}i32 @__kmpc_get_hardware_thread_id_in_block()
// SPIRV-NEXT:    [[WS0:%.*]] = call {{.*}}i32 @__kmpc_get_warp_size()
// SPIRV-NEXT:    %nvptx_lane_id = urem i32 [[TID0]], [[WS0]]
// SPIRV-NEXT:    [[TID1:%.*]] = call {{.*}}i32 @__kmpc_get_hardware_thread_id_in_block()
// SPIRV-NEXT:    [[WS1:%.*]] = call {{.*}}i32 @__kmpc_get_warp_size()
// SPIRV-NEXT:    %nvptx_warp_id = udiv i32 [[TID1]], [[WS1]]

// NVPTX-LABEL: define internal void @_omp_reduction_inter_warp_copy_func(
// NVPTX:         call i32 @__kmpc_get_hardware_thread_id_in_block()
// NVPTX-NEXT:    [[TID0:%.*]] = call i32 @__kmpc_get_hardware_thread_id_in_block()
// NVPTX-NEXT:    %nvptx_lane_id = and i32 [[TID0]], 31
// NVPTX-NEXT:    [[TID1:%.*]] = call i32 @__kmpc_get_hardware_thread_id_in_block()
// NVPTX-NEXT:    %nvptx_warp_id = ashr i32 [[TID1]], 5

int sum(int *a, int n) {
  int s = 0;
#pragma omp target teams distribute parallel for reduction(+ : s) map(to : a[:n])
  for (int i = 0; i < n; ++i)
    s += a[i];
  return s;
}
