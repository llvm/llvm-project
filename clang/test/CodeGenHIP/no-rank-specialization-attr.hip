// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1300 -x hip -emit-llvm -fcuda-is-device \
// RUN:   -o - %s | FileCheck %s

// Check that the amdgpu-no-rank-specialization IR attribute is properly generated from a HIP program.

#define __global__ __attribute__((global))
#define __exp_amd_wavegroup_kernel(num_wavegroups, wave_size, block_x, block_y, block_z) \
    __attribute__((amdgpu_wavegroup_kernel(num_wavegroups, wave_size, block_x, block_y, block_z)))
#define __exp_amd_no_rank_specialization__ __attribute__((amdgpu_no_rank_specialization))

__exp_amd_wavegroup_kernel(
    /* num_wavegroups */ 4,
    /* wave size */ 32,
    /* block size */ 16, 16, 1)
__exp_amd_no_rank_specialization__
__global__
// CHECK-LABEL: define dso_local amdgpu_kernel void @_Z3foov(
// CHECK-SAME: ) #[[ATTR0:[0-9]+]] !reqd_work_group_size ![[META0:[0-9]+]] {
void foo() {
}

// CHECK: attributes #[[ATTR0]] =
// CHECK-SAME: "amdgpu-no-rank-specialization"
