// RUN: %clang_cc1 -triple amdgpu9.00-amd-amdhsa -x hip -fcuda-is-device -emit-llvm -o - %s | FileCheck --check-prefix=DEV %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -aux-triple amdgpu9.00-amd-amdhsa -x hip -emit-llvm -o - %s | FileCheck --check-prefix=HOST %s
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -aux-triple amdgpu9.00-amd-amdhsa -x hip -emit-llvm -o - %s | FileCheck --check-prefix=HOST %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fcuda-is-device -emit-llvm -o - %s | FileCheck --check-prefix=DEV %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -aux-triple spirv64-amd-amdhsa -x hip -emit-llvm -o - %s | FileCheck --check-prefix=HOST %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -emit-llvm -o - %s | FileCheck --check-prefix=DEV %s
// RUN: echo -n "GPU binary would be here." > %t
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -aux-triple nvptx64-nvidia-cuda -fcuda-include-gpubinary %t -emit-llvm -o - %s | FileCheck --check-prefix=HOST %s

// The host registers a kernel with its device name, so both sides must mangle
// it in the same way. A language address space is mangled by its name, and a
// numeric one by its number.

#define __global__ __attribute__((global))

// DEV: define {{.*}}@_Z6kernelPU8CLglobali(
// HOST: c"_Z6kernelPU8CLglobali\00"
__global__ void kernel(__attribute__((opencl_global)) int *p) {}

// DEV: define {{.*}}@_Z7nkernelPU3AS1i(
// HOST: c"_Z7nkernelPU3AS1i\00"
__global__ void nkernel(__attribute__((address_space(1))) int *p) {}
