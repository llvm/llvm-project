// REQUIRES: comgr-has-spirv

// COM: Compile HIP source to SPIR-V using Comgr with additional options
// RUN: AMD_COMGR_EMIT_VERBOSE_LOGS=1 AMD_COMGR_REDIRECT_LOGS=source-to-spirv-logs.txt \
// RUN:   source-to-spirv %s %t.spv

// COM: Verify verbose logs show the SPIR-V target architecture
// RUN: grep 'offload-arch=amdgcnspirv' source-to-spirv-logs.txt

// COM: Verify the SPIR-V file was created and is non-empty
// RUN: test -s %t.spv

// COM: Translate SPIR-V back to LLVM IR bitcode
// RUN: spirv-translator %t.spv -o %t.bc

// COM: Disassemble LLVM IR bitcode to text and verify content
// RUN: llvm-dis %t.bc -o - | FileCheck %s

// COM: Verify LLVM IR contains expected functions, attributes, and target triple
// CHECK: target triple = "amdgcn-amd-amdhsa"
// CHECK: define void @_Z11clean_valuePf
// CHECK: define amdgpu_kernel void @_Z12complex_mathPfS_S_
// CHECK: call {{.*}} @llvm.sqrt
// CHECK: call {{.*}} @llvm.sin

// RUN: rm source-to-spirv-logs.txt

#include <cstdlib>
#include <cmath>

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
void complex_math(float* a, float* b, float* res) {
    float temp = sqrt(*a) + sin(*b);
    *res = temp * 2.0f;

    clean_value(a);
}

