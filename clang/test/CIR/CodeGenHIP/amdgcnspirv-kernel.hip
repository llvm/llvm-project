// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefixes=CIR,CIR-DEFAULT %s --input-file=%t.cir

// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-DEFAULT %s --input-file=%t.ll

// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fclangir \
// RUN:            --gpu-max-threads-per-block=256 \
// RUN:            -fcuda-is-device -emit-cir %s -o %t.max256.cir
// RUN: FileCheck --check-prefixes=CIR,CIR-MAX256 %s --input-file=%t.max256.cir

// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fclangir \
// RUN:            --gpu-max-threads-per-block=256 \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.max256.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-MAX256 %s --input-file=%t.max256.ll

// RUN: %clang_cc1 -triple spirv64 -x hip -fclangir \
// RUN:            -fcuda-is-device -emit-cir %s -o %t.noamd.cir
// RUN: FileCheck --check-prefix=NOAMD %s --input-file=%t.noamd.cir

// RUN: %clang_cc1 -triple spirv64 -x hip -fclangir \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.noamd.ll
// RUN: FileCheck --check-prefix=NOAMD %s --input-file=%t.noamd.ll

// RUN: %clang_cc1 -triple spir64-amd-unknown -x hip -fclangir \
// RUN:            -fcuda-is-device -emit-cir %s -o %t.spir.cir
// RUN: FileCheck --check-prefix=NOAMD %s --input-file=%t.spir.cir

// Test that HIP kernels on AMDGCN-flavored SPIR-V get the spir_kernel
// calling convention.

// CIR: cir.cl.version = #cir.cl.version<2, 0>

// NOAMD-NOT: max_work_group_size

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))

// CIR: cir.func{{.*}} @_Z13kernel_scalari{{.*}} cc(spir_kernel)
// CIR-DEFAULT-SAME: cir.max_work_group_size = #cir.max_work_group_size<1024, 1, 1>
// CIR-MAX256-SAME: cir.max_work_group_size = #cir.max_work_group_size<256, 1, 1>
// LLVM: define spir_kernel void @_Z13kernel_scalari{{.*}} !max_work_group_size [[MWGS_DEFAULT:![0-9]+]]
__global__ void kernel_scalar(int a) {}

// CIR: cir.func{{.*}} @_Z9device_fni
// CIR-NOT: cc(spir_kernel)
// CIR-NOT: cir.max_work_group_size
// LLVM: define{{.*}} void @_Z9device_fni
// LLVM-NOT: !max_work_group_size
__device__ void device_fn(int a) {}

// CIR: cir.func{{.*}} @_Z20flat_work_group_sizev{{.*}}cir.max_work_group_size = #cir.max_work_group_size<64, 1, 1>
// LLVM: define spir_kernel void @_Z20flat_work_group_sizev(){{.*}} !max_work_group_size [[MWGS_64:![0-9]+]]
__attribute__((amdgpu_flat_work_group_size(32, 64)))
__global__ void flat_work_group_size() {}

// CIR: cir.func{{.*}} @_Z13launch_boundsv{{.*}}cir.max_work_group_size = #cir.max_work_group_size<128, 1, 1>
// LLVM: define spir_kernel void @_Z13launch_boundsv(){{.*}} !max_work_group_size [[MWGS_128:![0-9]+]]
__launch_bounds__(128, 2)
__global__ void launch_bounds() {}

// CIR: cir.func{{.*}} @_Z18launch_bounds_zerov{{.*}} cc(spir_kernel)
// CIR-DEFAULT-SAME: cir.max_work_group_size = #cir.max_work_group_size<1024, 1, 1>
// CIR-MAX256-SAME: cir.max_work_group_size = #cir.max_work_group_size<256, 1, 1>
// LLVM: define spir_kernel void @_Z18launch_bounds_zerov(){{.*}} !max_work_group_size [[MWGS_DEFAULT]]
__launch_bounds__(0)
__global__ void launch_bounds_zero() {}

// CIR: cir.func{{.*}} @_Z24launch_bounds_overriddenv{{.*}}cir.max_work_group_size = #cir.max_work_group_size<32, 1, 1>
// LLVM: define spir_kernel void @_Z24launch_bounds_overriddenv(){{.*}} !max_work_group_size [[MWGS_32:![0-9]+]]
__attribute__((amdgpu_flat_work_group_size(8, 32)))
__launch_bounds__(128)
__global__ void launch_bounds_overridden() {}

// LLVM: !opencl.ocl.version = !{[[OCL:![0-9]+]]}
// LLVM-DAG: [[OCL]] = !{i32 2, i32 0}
// LLVM-DEFAULT-DAG: [[MWGS_DEFAULT]] = !{i32 1024, i32 1, i32 1}
// LLVM-MAX256-DAG: [[MWGS_DEFAULT]] = !{i32 256, i32 1, i32 1}
// LLVM-DAG: [[MWGS_64]] = !{i32 64, i32 1, i32 1}
// LLVM-DAG: [[MWGS_128]] = !{i32 128, i32 1, i32 1}
// LLVM-DAG: [[MWGS_32]] = !{i32 32, i32 1, i32 1}
