// REQUIRES: comgr-has-spirv-translator && comgr-has-spirv-backend

// COM: Validate -use-spirv-backend for SPIR-V codegen (needs LLVM SPIR-V target).

// COM: === Setting 1: source-to-spirv -nogpuinc -use-spirv-backend ===
// RUN: source-to-spirv -nogpuinc -use-spirv-backend %s %t.spv

// COM: Verify the SPIR-V file was created and is non-empty
// RUN: test -s %t.spv

// COM: Translate SPIR-V back to LLVM IR bitcode
// RUN: spirv-translator %t.spv -o %t.bc

// COM: Disassemble LLVM IR bitcode to text
// RUN: %llvm-dis %t.bc -o - | %FileCheck %s --check-prefixes=CHECK,SPIRVBE

// COM: === Setting 2: source-to-spirv -nogpuinc -c -use-spirv-backend ===
// RUN: source-to-spirv -nogpuinc -c -use-spirv-backend %s %t.spv

// COM: Verify the SPIR-V file was created and is non-empty
// RUN: test -s %t.spv

// COM: Translate SPIR-V back to LLVM IR bitcode
// RUN: spirv-translator %t.spv -o %t.bc

// COM: Disassemble LLVM IR bitcode to text
// RUN: %llvm-dis %t.bc -o - | %FileCheck %s --check-prefixes=CHECK,SPIRVBE

// COM: Verify LLVM IR contains expected functions and target triple
// CHECK: target triple = "amdgcn-amd-amdhsa"
// SPIRVBE: define void @_Z11clean_valuePf
// CHECK: define amdgpu_kernel void @_Z9add_valuePfS_S_

__attribute__((device))
void clean_value(float* ptr) { *ptr = 0; }

__attribute__((global))
void add_value(float* a, float* b, float* res) {
    *res = *a + *b;

    clean_value(a);
}
