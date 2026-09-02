// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa \
// RUN:     -fcuda-is-device -emit-llvm -o - -x hip %s \
// RUN:     | FileCheck -check-prefixes=CHECK,DEFAULT %s
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa --gpu-max-threads-per-block=1024 \
// RUN:     -fcuda-is-device -emit-llvm -o - -x hip %s \
// RUN:     | FileCheck -check-prefixes=CHECK,MAX1024 %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa --gpu-max-threads-per-block=1024 \
// RUN:     -fcuda-is-device -emit-llvm -o - -x hip %s \
// RUN:     | FileCheck -check-prefixes=CHECK-SPIRV,MAX1024-SPIRV %s
// RUN: %clang_cc1 -triple nvptx \
// RUN:     -fcuda-is-device -emit-llvm -o - %s | FileCheck %s \
// RUN:     -check-prefix=NAMD
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm \
// RUN:     -verify -Wno-deprecated-declarations -o - -x hip %s | FileCheck -check-prefix=NAMD %s

// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -foffload-uniform-block \
// RUN:     -fcuda-is-device -emit-llvm -o - -x hip %s \
// RUN:     | FileCheck -check-prefixes=CHECK,DEFAULT %s
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -fno-offload-uniform-block \
// RUN:     -fcuda-is-device -emit-llvm -o - -x hip %s \
// RUN:     | FileCheck -check-prefixes=NOUB %s

#include "Inputs/cuda.h"

__global__ void flat_work_group_size_default() {
// CHECK: define{{.*}} amdgpu_kernel void @_Z28flat_work_group_size_defaultv() [[FLAT_WORK_GROUP_SIZE_DEFAULT:#[0-9]+]]
// CHECK-SPIRV: define{{.*}} spir_kernel void @_Z28flat_work_group_size_defaultv(){{.*}} !max_work_group_size [[MAX_WORK_GROUP_SIZE_DEFAULT:![0-9]+]]
// NOUB: define{{.*}} void @_Z28flat_work_group_size_defaultv() [[NOUB:#[0-9]+]]
}

__attribute__((amdgpu_flat_work_group_size(32, 64))) // expected-no-diagnostics
__global__ void flat_work_group_size_32_64() {
// CHECK: define{{.*}} amdgpu_kernel void @_Z26flat_work_group_size_32_64v() [[FLAT_WORK_GROUP_SIZE_32_64:#[0-9]+]]
// CHECK-SPIRV: define{{.*}} spir_kernel void @_Z26flat_work_group_size_32_64v(){{.*}} !max_work_group_size [[MAX_WORK_GROUP_SIZE_64:![0-9]+]]
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

template<unsigned a>
__attribute__((amdgpu_max_num_work_groups(a)))
__global__ void template_a_max_num_work_groups() {}
template __global__ void template_a_max_num_work_groups<32>();
// CHECK: define{{.*}} amdgpu_kernel void @_Z30template_a_max_num_work_groupsILj32EEvv() [[MAX_NUM_WORK_GROUPS_32_1_1]]

template<unsigned a, unsigned b>
__attribute__((amdgpu_max_num_work_groups(a, b)))
__global__ void template_a_b_max_num_work_groups() {}
template __global__ void template_a_b_max_num_work_groups<32, 1>();
// CHECK: define{{.*}} amdgpu_kernel void @_Z32template_a_b_max_num_work_groupsILj32ELj1EEvv() [[MAX_NUM_WORK_GROUPS_32_1_1]]

template<unsigned a, unsigned b, unsigned c>
__attribute__((amdgpu_max_num_work_groups(a, b, c)))
__global__ void template_a_b_c_max_num_work_groups() {}
template __global__ void template_a_b_c_max_num_work_groups<32, 4, 2>();
// CHECK: define{{.*}} amdgpu_kernel void @_Z34template_a_b_c_max_num_work_groupsILj32ELj4ELj2EEvv() [[MAX_NUM_WORK_GROUPS_32_4_2]]

// __launch_bounds__ is consumed directly on AMDGPU: the first argument maps to
// the maximum flat work group size and the (optional) second to the minimum
// waves per execution unit.
__launch_bounds__(128)
__global__ void launch_bounds_1arg() {
// CHECK: define{{.*}} amdgpu_kernel void @_Z18launch_bounds_1argv() [[LAUNCH_BOUNDS_1ARG:#[0-9]+]]
// CHECK-SPIRV: define{{.*}} spir_kernel void @_Z18launch_bounds_1argv(){{.*}} !max_work_group_size [[MAX_WORK_GROUP_SIZE_128:![0-9]+]]
}

__launch_bounds__(128, 2)
__global__ void launch_bounds_2arg() {
// CHECK: define{{.*}} amdgpu_kernel void @_Z18launch_bounds_2argv() [[LAUNCH_BOUNDS_2ARG:#[0-9]+]]
// CHECK-SPIRV: define{{.*}} spir_kernel void @_Z18launch_bounds_2argv(){{.*}} !max_work_group_size [[MAX_WORK_GROUP_SIZE_128]]
}

// The third argument (maxclusterrank) is not yet handled on AMDGPU; it is
// silently ignored without the NVPTX sm_90 diagnostic.
__launch_bounds__(128, 2, 4)
__global__ void launch_bounds_3arg() {
// CHECK: define{{.*}} amdgpu_kernel void @_Z18launch_bounds_3argv() [[LAUNCH_BOUNDS_2ARG]]
// CHECK-SPIRV: define{{.*}} spir_kernel void @_Z18launch_bounds_3argv(){{.*}} !max_work_group_size [[MAX_WORK_GROUP_SIZE_128]]
}

// An explicit amdgpu_flat_work_group_size / amdgpu_waves_per_eu takes precedence
// over __launch_bounds__.
__attribute__((amdgpu_flat_work_group_size(8, 32), amdgpu_waves_per_eu(4)))
__launch_bounds__(128, 2)
__global__ void launch_bounds_explicit_override() {
// CHECK: define{{.*}} amdgpu_kernel void @_Z31launch_bounds_explicit_overridev() [[LAUNCH_BOUNDS_OVERRIDE:#[0-9]+]]
// CHECK-SPIRV: define{{.*}} spir_kernel void @_Z31launch_bounds_explicit_overridev(){{.*}} !max_work_group_size [[MAX_WORK_GROUP_SIZE_32:![0-9]+]]
}

// The launch bounds from an earlier declaration are kept when the definition
// does not specify any.
__launch_bounds__(128, 2)
__global__ void launch_bounds_redecl_def_none();
__global__ void launch_bounds_redecl_def_none() {
// CHECK: define{{.*}} amdgpu_kernel void @_Z29launch_bounds_redecl_def_nonev() [[LAUNCH_BOUNDS_2ARG]]
}

// Launch bounds specified only on the definition are honored.
__global__ void launch_bounds_redecl_decl_none();
__launch_bounds__(128, 2)
__global__ void launch_bounds_redecl_decl_none() {
// CHECK: define{{.*}} amdgpu_kernel void @_Z30launch_bounds_redecl_decl_nonev() [[LAUNCH_BOUNDS_2ARG]]
}

// When multiple declarations specify conflicting launch bounds, the last one
// wins.
__launch_bounds__(64, 8)
__global__ void launch_bounds_redecl_conflict();
__launch_bounds__(128, 2)
__global__ void launch_bounds_redecl_conflict();
__global__ void launch_bounds_redecl_conflict() {
// CHECK: define{{.*}} amdgpu_kernel void @_Z29launch_bounds_redecl_conflictv() [[LAUNCH_BOUNDS_2ARG]]
}

// __launch_bounds__ only takes effect on kernels; it is silently ignored on
// __device__ functions.
__launch_bounds__(128, 2)
__device__ void launch_bounds_device_fn() {
// CHECK: define{{.*}} void @_Z23launch_bounds_device_fnv() [[LAUNCH_BOUNDS_DEVICE:#[0-9]+]]
}

// Make sure this is silently accepted on other targets.
// NAMD-NOT: "amdgpu-flat-work-group-size"
// NAMD-NOT: "amdgpu-waves-per-eu"
// NAMD-NOT: "amdgpu-num-vgpr"
// NAMD-NOT: "amdgpu-num-sgpr"
// NAMD-NOT: "amdgpu-max-num-work-groups"

// DEFAULT-DAG: attributes [[FLAT_WORK_GROUP_SIZE_DEFAULT]] = {{.*}}"amdgpu-flat-work-group-size"="1,1024"{{.*}}"uniform-work-group-size"
// MAX1024-DAG: attributes [[FLAT_WORK_GROUP_SIZE_DEFAULT]] = {{.*}}"amdgpu-flat-work-group-size"="1,1024"
// MAX1024-SPIRV-DAG: [[MAX_WORK_GROUP_SIZE_DEFAULT]] = !{i32 1024, i32 1, i32 1}
// CHECK-DAG: attributes [[FLAT_WORK_GROUP_SIZE_32_64]] = {{.*}}"amdgpu-flat-work-group-size"="32,64"
// CHECK-SPIRV-DAG: [[MAX_WORK_GROUP_SIZE_64]] = !{i32 64, i32 1, i32 1}
// CHECK-DAG: attributes [[WAVES_PER_EU_2]] = {{.*}}"amdgpu-waves-per-eu"="2"
// CHECK-DAG: attributes [[NUM_SGPR_32]] = {{.*}}"amdgpu-num-sgpr"="32"
// CHECK-DAG: attributes [[NUM_VGPR_64]] = {{.*}}"amdgpu-num-vgpr"="64"
// CHECK-DAG: attributes [[MAX_NUM_WORK_GROUPS_32_4_2]] = {{.*}}"amdgpu-max-num-workgroups"="32,4,2"
// CHECK-DAG: attributes [[MAX_NUM_WORK_GROUPS_32_1_1]] = {{.*}}"amdgpu-max-num-workgroups"="32,1,1"
// CHECK-DAG: attributes [[LAUNCH_BOUNDS_1ARG]] = {{.*}}"amdgpu-flat-work-group-size"="1,128"
// CHECK-DAG: attributes [[LAUNCH_BOUNDS_2ARG]] = {{.*}}"amdgpu-flat-work-group-size"="1,128"{{.*}}"amdgpu-waves-per-eu"="2"
// CHECK-DAG: attributes [[LAUNCH_BOUNDS_OVERRIDE]] = {{.*}}"amdgpu-flat-work-group-size"="8,32"{{.*}}"amdgpu-waves-per-eu"="4"
// __launch_bounds__ is ignored on __device__ functions, so no
// amdgpu-flat-work-group-size / amdgpu-waves-per-eu attribute is present.
// String attributes are sorted, so the amdgpu-* attributes would appear
// immediately after "optnone"; check that "no-trapping-math" follows directly.
// CHECK-DAG: attributes [[LAUNCH_BOUNDS_DEVICE]] = { convergent mustprogress noipa noinline nounwind optnone "no-trapping-math"={{.*}}"uniform-work-group-size" }
// CHECK-SPIRV-DAG: [[MAX_WORK_GROUP_SIZE_128]] = !{i32 128, i32 1, i32 1}
// CHECK-SPIRV-DAG: [[MAX_WORK_GROUP_SIZE_32]] = !{i32 32, i32 1, i32 1}

// NOUB-NOT: "uniform-work-group-size"
