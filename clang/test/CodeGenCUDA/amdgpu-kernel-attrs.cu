// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa \
// RUN:     -fcuda-is-device -emit-llvm -o - -x hip %s \
// RUN:     | FileCheck -check-prefixes=CHECK,DEFAULT %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa --gpu-max-threads-per-block=1024 \
// RUN:     -fcuda-is-device -emit-llvm -o - -x hip %s \
// RUN:     | FileCheck -check-prefixes=CHECK,MAX1024 %s
// RUN: %clang_cc1 -triple nvptx \
// RUN:     -fcuda-is-device -emit-llvm -o - %s | FileCheck %s \
// RUN:     -check-prefix=NAMD
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm \
// RUN:     -verify -o - -x hip %s | FileCheck -check-prefix=NAMD %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -foffload-uniform-block \
// RUN:     -fcuda-is-device -emit-llvm -o - -x hip %s \
// RUN:     | FileCheck -check-prefixes=CHECK,DEFAULT %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fno-offload-uniform-block \
// RUN:     -fcuda-is-device -emit-llvm -o - -x hip %s \
// RUN:     | FileCheck -check-prefixes=NOUB %s

#include "Inputs/cuda.h"

__global__ void flat_work_group_size_default() {
// CHECK: define{{.*}} amdgpu_kernel void @_Z28flat_work_group_size_defaultv() [[FLAT_WORK_GROUP_SIZE_DEFAULT:#[0-9]+]]
// NOUB: define{{.*}} void @_Z28flat_work_group_size_defaultv() [[NOUB:#[0-9]+]]
}

__attribute__((amdgpu_flat_work_group_size(32, 64))) // expected-no-diagnostics
__global__ void flat_work_group_size_32_64() {
// CHECK: define{{.*}} amdgpu_kernel void @_Z26flat_work_group_size_32_64v() [[FLAT_WORK_GROUP_SIZE_32_64:#[0-9]+]]
}
__attribute__((amdgpu_waves_per_eu(2))) // expected-no-diagnostics
__global__ void waves_per_eu_2() {
// CHECK: define{{.*}} amdgpu_kernel void @_Z14waves_per_eu_2v() [[WAVES_PER_EU_2:#[0-9]+]]
}
__attribute__((amdgpu_num_sgpr(32))) // expected-no-diagnostics
__global__ void num_sgpr_32() {
// CHECK: define{{.*}} amdgpu_kernel void @_Z11num_sgpr_32v() [[NUM_SGPR_32:#[0-9]+]]
}
__attribute__((amdgpu_num_vgpr(64))) // expected-no-diagnostics
__global__ void num_vgpr_64() {
// CHECK: define{{.*}} amdgpu_kernel void @_Z11num_vgpr_64v() [[NUM_VGPR_64:#[0-9]+]]
}
__attribute__((amdgpu_max_num_work_groups(32, 4, 2))) // expected-no-diagnostics
__global__ void max_num_work_groups_32_4_2() {
// CHECK: define{{.*}} amdgpu_kernel void @_Z26max_num_work_groups_32_4_2v() [[MAX_NUM_WORK_GROUPS_32_4_2:#[0-9]+]]
}
__attribute__((amdgpu_max_num_work_groups(32))) // expected-no-diagnostics
__global__ void max_num_work_groups_32() {
// CHECK: define{{.*}} amdgpu_kernel void @_Z22max_num_work_groups_32v() [[MAX_NUM_WORK_GROUPS_32_1_1:#[0-9]+]]
}
__attribute__((amdgpu_max_num_work_groups(32,1))) // expected-no-diagnostics
__global__ void max_num_work_groups_32_1() {
// CHECK: define{{.*}} amdgpu_kernel void @_Z24max_num_work_groups_32_1v() [[MAX_NUM_WORK_GROUPS_32_1_1:#[0-9]+]]
}



template<unsigned a>
__attribute__((amdgpu_max_num_work_groups(a, 4, 2)))
__global__ void template_a_4_2_max_num_work_groups() {}
template __global__ void template_a_4_2_max_num_work_groups<32>();
// CHECK: define{{.*}} amdgpu_kernel void @_Z34template_a_4_2_max_num_work_groupsILj32EEvv() [[MAX_NUM_WORK_GROUPS_32_4_2:#[0-9]+]]

template<unsigned a>
__attribute__((amdgpu_max_num_work_groups(32, a, 2)))
__global__ void template_32_a_2_max_num_work_groups() {}
template __global__ void template_32_a_2_max_num_work_groups<4>();
// CHECK: define{{.*}} amdgpu_kernel void @_Z35template_32_a_2_max_num_work_groupsILj4EEvv() [[MAX_NUM_WORK_GROUPS_32_4_2:#[0-9]+]]

template<unsigned a>
__attribute__((amdgpu_max_num_work_groups(32, 4, a)))
__global__ void template_32_4_a_max_num_work_groups() {}
template __global__ void template_32_4_a_max_num_work_groups<2>();
// CHECK: define{{.*}} amdgpu_kernel void @_Z35template_32_4_a_max_num_work_groupsILj2EEvv() [[MAX_NUM_WORK_GROUPS_32_4_2:#[0-9]+]]

// Make sure this is silently accepted on other targets.
// NAMD-NOT: "amdgpu-flat-work-group-size"
// NAMD-NOT: "amdgpu-waves-per-eu"
// NAMD-NOT: "amdgpu-num-vgpr"
// NAMD-NOT: "amdgpu-num-sgpr"
// NAMD-NOT: "amdgpu-max-num-work-groups"

// DEFAULT-DAG: attributes [[FLAT_WORK_GROUP_SIZE_DEFAULT]] = {{.*}}"amdgpu-flat-work-group-size"="1,1024"{{.*}}"uniform-work-group-size"="true"
// MAX1024-DAG: attributes [[FLAT_WORK_GROUP_SIZE_DEFAULT]] = {{.*}}"amdgpu-flat-work-group-size"="1,1024"
// CHECK-DAG: attributes [[FLAT_WORK_GROUP_SIZE_32_64]] = {{.*}}"amdgpu-flat-work-group-size"="32,64"
// CHECK-DAG: attributes [[WAVES_PER_EU_2]] = {{.*}}"amdgpu-waves-per-eu"="2"
// CHECK-DAG: attributes [[NUM_SGPR_32]] = {{.*}}"amdgpu-num-sgpr"="32"
// CHECK-DAG: attributes [[NUM_VGPR_64]] = {{.*}}"amdgpu-num-vgpr"="64"
// CHECK-DAG: attributes [[MAX_NUM_WORK_GROUPS_32_4_2]] = {{.*}}"amdgpu-max-num-workgroups"="32,4,2"
// CHECK-DAG: attributes [[MAX_NUM_WORK_GROUPS_32_1_1]] = {{.*}}"amdgpu-max-num-workgroups"="32,1,1"

// NOUB-NOT: "uniform-work-group-size"="true"
