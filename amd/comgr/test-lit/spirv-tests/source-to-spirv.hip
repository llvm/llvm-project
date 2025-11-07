// REQUIRES: comgr-has-spirv

// COM: Compile HIP source to SPIR-V using Comgr
// RUN: source-to-spirv %s %t.spv

// COM: Verify the SPIR-V file was created and is non-empty
// RUN: test -s %t.spv

// COM: Translate SPIR-V back to LLVM IR bitcode
// RUN: spirv-translator %t.spv -o %t.bc

// COM: Disassemble LLVM IR bitcode to text
// RUN: llvm-dis %t.bc -o - | FileCheck %s

// COM: Verify LLVM IR contains expected functions and target triple
// CHECK: target triple = "amdgcn-amd-amdhsa"
// CHECK: define void @_Z11clean_valuePf
// CHECK: define amdgpu_kernel void @_Z9add_valuePfS_S_

#include <cstdlib>

#define __constant__ __attribute__((constant))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))
#define __managed__ __attribute__((managed))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))

__attribute__((device))
void clean_value(float* ptr) { *ptr = 0; }

__attribute__((global))
void add_value(float* a, float* b, float* res) {
    *res = *a + *b;

    clean_value(a);
}

