// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1300 -x hip -emit-llvm -fcuda-is-device \
// RUN:   -o - %s | FileCheck %s

// Check that the amdgpu-wavegroup-enable and amdgpu-workgroup-size
// IR attributes are generated correctly.

#define __global__ __attribute__((global))
#define __exp_amd_wavegroup_kernel(num_wavegroups, wave_size, block_x, block_y, block_z) \
    __attribute__((amdgpu_wavegroup_kernel(num_wavegroups, wave_size, block_x, block_y, block_z)))

__exp_amd_wavegroup_kernel(
    /* num_wavegroups */ 4,
    /* wave size */ 32,
    /* block size */ 32, 8, 1)
__global__
// CHECK-LABEL: define dso_local amdgpu_kernel void @_Z3foov(
// CHECK-SAME: ) #[[ATTR0:[0-9]+]] !reqd_work_group_size ![[META0:[0-9]+]] {
void foo() {
}

// CHECK: attributes #[[ATTR0]] =
// CHECK-SAME: "amdgpu-wavegroup-enable"

// CHECK: ![[META0]] = !{i32 32, i32 8, i32 1}
