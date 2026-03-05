#include "../Inputs/cuda.h"

// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR %s --input-file=%t.cir

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM %s --input-file=%t.ll

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ogcg.ll
// RUN: FileCheck --check-prefix=OGCG %s --input-file=%t.ogcg.ll

// Test that AMDGPU-specific attributes are generated for HIP kernels

// Check CIR attribute definitions
// CIR-DAG: "amdgpu-flat-work-group-size" = "1,1024"
// CIR-DAG: "amdgpu-flat-work-group-size" = "64,128"
// CIR-DAG: "amdgpu-waves-per-eu" = "2"
// CIR-DAG: "amdgpu-waves-per-eu" = "2,4"
// CIR-DAG: "amdgpu-num-sgpr" = "32"
// CIR-DAG: "amdgpu-num-vgpr" = "64"
// CIR-DAG: "amdgpu-max-num-workgroups" = "8,4,2"
// CIR-DAG: "amdgpu-flat-work-group-size" = "256,256"{{.*}}"amdgpu-num-sgpr" = "48"{{.*}}"amdgpu-num-vgpr" = "32"{{.*}}"amdgpu-waves-per-eu" = "1,2"

// Test: Default attributes for simple kernel
// CIR: cir.func{{.*}} @_Z13kernel_simplev() cc(amdgpu_kernel)
// LLVM: define{{.*}} amdgpu_kernel void @_Z13kernel_simplev(){{.*}} #[[SIMPLE_ATTR:[0-9]+]]
// OGCG: define{{.*}} amdgpu_kernel void @_Z13kernel_simplev(){{.*}} #[[OGCG_SIMPLE_ATTR:[0-9]+]]
__global__ void kernel_simple() {}

// Test: Explicit flat work group size attribute
// CIR: cir.func{{.*}} @_Z21kernel_flat_wg_size_1v() cc(amdgpu_kernel)
// LLVM: define{{.*}} amdgpu_kernel void @_Z21kernel_flat_wg_size_1v(){{.*}} #[[FLAT_WG_ATTR:[0-9]+]]
// OGCG: define{{.*}} amdgpu_kernel void @_Z21kernel_flat_wg_size_1v(){{.*}} #[[OGCG_FLAT_WG_ATTR:[0-9]+]]
__attribute__((amdgpu_flat_work_group_size(64, 128)))
__global__ void kernel_flat_wg_size_1() {}

// Test: Waves per EU attribute
// CIR: cir.func{{.*}} @_Z19kernel_waves_per_euv() cc(amdgpu_kernel)
// LLVM: define{{.*}} amdgpu_kernel void @_Z19kernel_waves_per_euv(){{.*}} #[[WAVES_ATTR:[0-9]+]]
// OGCG: define{{.*}} amdgpu_kernel void @_Z19kernel_waves_per_euv(){{.*}} #[[OGCG_WAVES_ATTR:[0-9]+]]
__attribute__((amdgpu_waves_per_eu(2)))
__global__ void kernel_waves_per_eu() {}

// Test: Waves per EU with min and max
// CIR: cir.func{{.*}} @_Z22kernel_waves_per_eu_mmv() cc(amdgpu_kernel)
// LLVM: define{{.*}} amdgpu_kernel void @_Z22kernel_waves_per_eu_mmv(){{.*}} #[[WAVES_MM_ATTR:[0-9]+]]
// OGCG: define{{.*}} amdgpu_kernel void @_Z22kernel_waves_per_eu_mmv(){{.*}} #[[OGCG_WAVES_MM_ATTR:[0-9]+]]
__attribute__((amdgpu_waves_per_eu(2, 4)))
__global__ void kernel_waves_per_eu_mm() {}

// Test: Num SGPR attribute
// CIR: cir.func{{.*}} @_Z15kernel_num_sgprv() cc(amdgpu_kernel)
// LLVM: define{{.*}} amdgpu_kernel void @_Z15kernel_num_sgprv(){{.*}} #[[SGPR_ATTR:[0-9]+]]
// OGCG: define{{.*}} amdgpu_kernel void @_Z15kernel_num_sgprv(){{.*}} #[[OGCG_SGPR_ATTR:[0-9]+]]
__attribute__((amdgpu_num_sgpr(32)))
__global__ void kernel_num_sgpr() {}

// Test: Num VGPR attribute
// CIR: cir.func{{.*}} @_Z15kernel_num_vgprv() cc(amdgpu_kernel)
// LLVM: define{{.*}} amdgpu_kernel void @_Z15kernel_num_vgprv(){{.*}} #[[VGPR_ATTR:[0-9]+]]
// OGCG: define{{.*}} amdgpu_kernel void @_Z15kernel_num_vgprv(){{.*}} #[[OGCG_VGPR_ATTR:[0-9]+]]
__attribute__((amdgpu_num_vgpr(64)))
__global__ void kernel_num_vgpr() {}

// Test: Max num workgroups attribute
// CIR: cir.func{{.*}} @_Z22kernel_max_num_wgroupsv() cc(amdgpu_kernel)
// LLVM: define{{.*}} amdgpu_kernel void @_Z22kernel_max_num_wgroupsv(){{.*}} #[[MAX_WG_ATTR:[0-9]+]]
// OGCG: define{{.*}} amdgpu_kernel void @_Z22kernel_max_num_wgroupsv(){{.*}} #[[OGCG_MAX_WG_ATTR:[0-9]+]]
__attribute__((amdgpu_max_num_work_groups(8, 4, 2)))
__global__ void kernel_max_num_wgroups() {}

// Test: Combined attributes
// CIR: cir.func{{.*}} @_Z15kernel_combinedv() cc(amdgpu_kernel)
// LLVM: define{{.*}} amdgpu_kernel void @_Z15kernel_combinedv(){{.*}} #[[COMBINED_ATTR:[0-9]+]]
// OGCG: define{{.*}} amdgpu_kernel void @_Z15kernel_combinedv(){{.*}} #[[OGCG_COMBINED_ATTR:[0-9]+]]
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__attribute__((amdgpu_waves_per_eu(1, 2)))
__attribute__((amdgpu_num_sgpr(48)))
__attribute__((amdgpu_num_vgpr(32)))
__global__ void kernel_combined() {}

// Test: Device function should NOT have kernel attributes
// CIR: cir.func{{.*}} @_Z9device_fnv()
// CIR-NOT: cc(amdgpu_kernel)
// LLVM: define{{.*}} void @_Z9device_fnv()
// LLVM-NOT: amdgpu_kernel
// OGCG: define{{.*}} void @_Z9device_fnv()
// OGCG-NOT: amdgpu_kernel
__device__ void device_fn() {}

// Verify LLVM attributes
// LLVM-DAG: attributes #[[SIMPLE_ATTR]] = {{.*}}"amdgpu-flat-work-group-size"="1,1024"
// LLVM-DAG: attributes #[[FLAT_WG_ATTR]] = {{.*}}"amdgpu-flat-work-group-size"="64,128"
// LLVM-DAG: attributes #[[WAVES_ATTR]] = {{.*}}"amdgpu-waves-per-eu"="2"
// LLVM-DAG: attributes #[[WAVES_MM_ATTR]] = {{.*}}"amdgpu-waves-per-eu"="2,4"
// LLVM-DAG: attributes #[[SGPR_ATTR]] = {{.*}}"amdgpu-num-sgpr"="32"
// LLVM-DAG: attributes #[[VGPR_ATTR]] = {{.*}}"amdgpu-num-vgpr"="64"
// LLVM-DAG: attributes #[[MAX_WG_ATTR]] = {{.*}}"amdgpu-max-num-workgroups"="8,4,2"
// LLVM-DAG: attributes #[[COMBINED_ATTR]] = {{.*}}"amdgpu-flat-work-group-size"="256,256"{{.*}}"amdgpu-num-sgpr"="48"{{.*}}"amdgpu-num-vgpr"="32"{{.*}}"amdgpu-waves-per-eu"="1,2"

// Verify OGCG attributes
// OGCG-DAG: attributes #[[OGCG_SIMPLE_ATTR]] = {{.*}}"amdgpu-flat-work-group-size"="1,1024"
// OGCG-DAG: attributes #[[OGCG_FLAT_WG_ATTR]] = {{.*}}"amdgpu-flat-work-group-size"="64,128"
// OGCG-DAG: attributes #[[OGCG_WAVES_ATTR]] = {{.*}}"amdgpu-waves-per-eu"="2"
// OGCG-DAG: attributes #[[OGCG_WAVES_MM_ATTR]] = {{.*}}"amdgpu-waves-per-eu"="2,4"
// OGCG-DAG: attributes #[[OGCG_SGPR_ATTR]] = {{.*}}"amdgpu-num-sgpr"="32"
// OGCG-DAG: attributes #[[OGCG_VGPR_ATTR]] = {{.*}}"amdgpu-num-vgpr"="64"
// OGCG-DAG: attributes #[[OGCG_MAX_WG_ATTR]] = {{.*}}"amdgpu-max-num-workgroups"="8,4,2"
// OGCG-DAG: attributes #[[OGCG_COMBINED_ATTR]] = {{.*}}"amdgpu-flat-work-group-size"="256,256"{{.*}}"amdgpu-num-sgpr"="48"{{.*}}"amdgpu-num-vgpr"="32"{{.*}}"amdgpu-waves-per-eu"="1,2"
