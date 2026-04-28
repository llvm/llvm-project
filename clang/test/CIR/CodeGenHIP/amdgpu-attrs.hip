#include "../CodeGenCUDA/Inputs/cuda.h"

// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip -fclangir \
// RUN:            -Wno-deprecated-declarations \
// RUN:            -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR %s --input-file=%t.cir

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip -fclangir \
// RUN:            -Wno-deprecated-declarations \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM %s --input-file=%t.ll

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip \
// RUN:            -Wno-deprecated-declarations \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ogcg.ll
// RUN: FileCheck --check-prefix=LLVM %s --input-file=%t.ogcg.ll

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip -fclangir \
// RUN:            -Wno-deprecated-declarations \
// RUN:            -fcuda-is-device -mamdgpu-expand-waitcnt-profiling \
// RUN:            -emit-cir %s -o %t.prof.cir
// RUN: FileCheck --check-prefix=PROF %s --input-file=%t.prof.cir

// Test that AMDGPU-specific attributes are generated for HIP kernels

// Test: Default attributes for simple kernel
// CIR: cir.func{{.*}} @_Z13kernel_simplev(){{.*}}"cir.amdgpu-flat-work-group-size" = "1,1024"
// LLVM: define{{.*}} void @_Z13kernel_simplev(){{.*}} #[[SIMPLE_ATTR:[0-9]+]]
__global__ void kernel_simple() {}

// Test: Explicit flat work group size attribute
// CIR: cir.func{{.*}} @_Z21kernel_flat_wg_size_1v(){{.*}}"cir.amdgpu-flat-work-group-size" = "64,128"
// LLVM: define{{.*}} void @_Z21kernel_flat_wg_size_1v(){{.*}} #[[FLAT_WG_ATTR:[0-9]+]]
__attribute__((amdgpu_flat_work_group_size(64, 128)))
__global__ void kernel_flat_wg_size_1() {}

// Test: Waves per EU attribute
// CIR: cir.func{{.*}} @_Z19kernel_waves_per_euv(){{.*}}"cir.amdgpu-waves-per-eu" = "2"
// LLVM: define{{.*}} void @_Z19kernel_waves_per_euv(){{.*}} #[[WAVES_ATTR:[0-9]+]]
__attribute__((amdgpu_waves_per_eu(2)))
__global__ void kernel_waves_per_eu() {}

// Test: Waves per EU with min and max
// CIR: cir.func{{.*}} @_Z22kernel_waves_per_eu_mmv(){{.*}}"cir.amdgpu-waves-per-eu" = "2,4"
// LLVM: define{{.*}} void @_Z22kernel_waves_per_eu_mmv(){{.*}} #[[WAVES_MM_ATTR:[0-9]+]]
__attribute__((amdgpu_waves_per_eu(2, 4)))
__global__ void kernel_waves_per_eu_mm() {}

// Test: Num SGPR attribute
// CIR: cir.func{{.*}} @_Z15kernel_num_sgprv(){{.*}}"cir.amdgpu-num-sgpr" = "32"
// LLVM: define{{.*}} void @_Z15kernel_num_sgprv(){{.*}} #[[SGPR_ATTR:[0-9]+]]
__attribute__((amdgpu_num_sgpr(32)))
__global__ void kernel_num_sgpr() {}

// Test: Num VGPR attribute
// CIR: cir.func{{.*}} @_Z15kernel_num_vgprv(){{.*}}"cir.amdgpu-num-vgpr" = "64"
// LLVM: define{{.*}} void @_Z15kernel_num_vgprv(){{.*}} #[[VGPR_ATTR:[0-9]+]]
__attribute__((amdgpu_num_vgpr(64)))
__global__ void kernel_num_vgpr() {}

// Test: Max num workgroups attribute
// CIR: cir.func{{.*}} @_Z22kernel_max_num_wgroupsv(){{.*}}"cir.amdgpu-max-num-workgroups" = "8,4,2"
// LLVM: define{{.*}} void @_Z22kernel_max_num_wgroupsv(){{.*}} #[[MAX_WG_ATTR:[0-9]+]]
__attribute__((amdgpu_max_num_work_groups(8, 4, 2)))
__global__ void kernel_max_num_wgroups() {}

// Test: Combined attributes
// CIR: cir.func{{.*}} @_Z15kernel_combinedv(){{.*}}"cir.amdgpu-flat-work-group-size" = "256,256"{{.*}}"cir.amdgpu-num-sgpr" = "48"{{.*}}"cir.amdgpu-num-vgpr" = "32"{{.*}}"cir.amdgpu-waves-per-eu" = "1,2"
// LLVM: define{{.*}} void @_Z15kernel_combinedv(){{.*}} #[[COMBINED_ATTR:[0-9]+]]
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__attribute__((amdgpu_waves_per_eu(1, 2)))
__attribute__((amdgpu_num_sgpr(48)))
__attribute__((amdgpu_num_vgpr(32)))
__global__ void kernel_combined() {}

// Test: Device function should NOT have kernel attributes
// CIR: cir.func{{.*}} @_Z9device_fnv()
// CIR-NOT: cir.amdgpu-flat-work-group-size
// LLVM: define{{.*}} void @_Z9device_fnv()
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

// Test: amdgpu-expand-waitcnt-profiling is set on all functions when enabled
// PROF: cir.func{{.*}} @_Z13kernel_simplev(){{.*}}"cir.amdgpu-expand-waitcnt-profiling"
// PROF: cir.func{{.*}} @_Z21kernel_flat_wg_size_1v(){{.*}}"cir.amdgpu-expand-waitcnt-profiling"
// PROF: cir.func{{.*}} @_Z19kernel_waves_per_euv(){{.*}}"cir.amdgpu-expand-waitcnt-profiling"
// PROF: cir.func{{.*}} @_Z22kernel_waves_per_eu_mmv(){{.*}}"cir.amdgpu-expand-waitcnt-profiling"
// PROF: cir.func{{.*}} @_Z15kernel_num_sgprv(){{.*}}"cir.amdgpu-expand-waitcnt-profiling"
// PROF: cir.func{{.*}} @_Z15kernel_num_vgprv(){{.*}}"cir.amdgpu-expand-waitcnt-profiling"
// PROF: cir.func{{.*}} @_Z22kernel_max_num_wgroupsv(){{.*}}"cir.amdgpu-expand-waitcnt-profiling"
// PROF: cir.func{{.*}} @_Z15kernel_combinedv(){{.*}}"cir.amdgpu-expand-waitcnt-profiling"
// PROF: cir.func{{.*}} @_Z9device_fnv(){{.*}}"cir.amdgpu-expand-waitcnt-profiling"
