// RUN: %clang_cc1 -triple spirv64 -x hip -emit-llvm -fcuda-is-device \
// RUN:   -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -emit-llvm -fcuda-is-device \
// RUN:   -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -emit-llvm -fcuda-is-device \
// RUN:   -o - %s | FileCheck %s --check-prefix=AMDGCN

#define __global__ __attribute__((global))

// CHECK: define {{.*}}spir_kernel void @_Z3fooPff(ptr addrspace(1) {{.*}}, float {{.*}})
__global__ void foo(float *a, float b) {
  *a = b;
}

// CHECK: !opencl.ocl.version = !{[[OCL:![0-9]+]]}
// CHECK: [[OCL]] = !{i32 2, i32 0}
// AMDGCN-NOT: !opencl.ocl.version
