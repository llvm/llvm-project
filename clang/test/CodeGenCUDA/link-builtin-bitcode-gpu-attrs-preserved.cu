// Verify the behavior of the +gfxN-insts in the way that
// rocm-device-libs should be built with. e.g. If the device libraries has a function
// with "+gfx11-insts", that attribute should still be present after linking and not
// overwritten with the current target's settings.

// This is important because at this time, many device-libs functions that are only
// available on some GPUs put an attribute such as "+gfx11-insts" so that
// AMDGPURemoveIncompatibleFunctions can detect & remove them if needed.

// Build the fake device library in the way rocm-device-libs should be built.
//
// RUN: %clang_cc1 -x cl -triple amdgpu-amd-amdhsa\
// RUN:   -mcode-object-version=none -emit-llvm-bc \
// RUN:   %S/Inputs/ocml-sample-target-attrs.cl -o %t.bc

// Check the default behavior
// RUN: %clang_cc1 -x hip -triple amdgpu8.03-amd-amdhsa -fcuda-is-device \
// RUN:   -mlink-builtin-bitcode %t.bc \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,INTERNALIZE

// RUN: %clang_cc1 -x hip -triple amdgpu11.01-amd-amdhsa -fcuda-is-device \
// RUN:   -mlink-builtin-bitcode %t.bc -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,INTERNALIZE

// Check the case where no internalization is performed
// RUN: %clang_cc1 -x hip -triple amdgpu8.03-amd-amdhsa \
// RUN:   -fcuda-is-device -mlink-bitcode-file %t.bc -emit-llvm %s -o -  | FileCheck %s --check-prefixes=CHECK,NOINTERNALIZE

// Check the case where no internalization is performed
// RUN: %clang_cc1 -x hip -triple amdgpu11.01-amd-amdhsa \
// RUN:   -fcuda-is-device -mlink-bitcode-file %t.bc -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,NOINTERNALIZE


// CHECK: define {{.*}} i64 @do_intrin_stuff() #[[ATTR:[0-9]+]]
// INTERNALIZE: attributes #[[ATTR]] = {{.*}} "target-features"="{{.*}}+gfx11-insts{{.*}}"
// NOINTERNALIZE: attributes #[[ATTR]] = {{.*}} "target-features"="+gfx11-insts"

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))

typedef unsigned long ulong;

extern "C" {
__device__ ulong do_intrin_stuff(void);

__global__ void kernel_f16(ulong* out) {
    *out = do_intrin_stuff();
  }
}
