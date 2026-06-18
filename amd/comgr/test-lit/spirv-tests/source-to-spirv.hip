// REQUIRES: comgr-has-spirv-translator
// COM: Validate flags for SPIR-V codegen via the translator path (no LLVM SPIR-V backend).

// COM: === Setting 1: source-to-spirv -nogpuinc (link / default output) ===
// RUN: source-to-spirv -nogpuinc %s %t.spv

// COM: Verify the SPIR-V file was created and is non-empty
// RUN: test -s %t.spv

// COM: Translate SPIR-V back to LLVM IR bitcode
// RUN: spirv-translator %t.spv -o %t.bc

// COM: Disassemble LLVM IR bitcode to text
// RUN: %llvm-dis %t.bc -o - | %FileCheck %s --check-prefixes=CHECK,TRANSLATOR

// COM: === Setting 2: source-to-spirv -nogpuinc -c (compile only) ===
// RUN: source-to-spirv -nogpuinc -c %s %t.spv

// COM: Verify the SPIR-V file was created and is non-empty
// RUN: test -s %t.spv

// COM: Translate SPIR-V back to LLVM IR bitcode
// RUN: spirv-translator %t.spv -o %t.bc

// COM: Disassemble LLVM IR bitcode to text
// RUN: %llvm-dis %t.bc -o - | %FileCheck %s --check-prefixes=CHECK,TRANSLATOR

// COM: Verify LLVM IR contains expected functions and target triple
// CHECK: target triple = "amdgcn-amd-amdhsa"
// TRANSLATOR: define void @_Z11clean_valuePf
// CHECK: define amdgpu_kernel void @_Z9add_valuePfS_S_

__attribute__((device))
void clean_value(float* ptr) { *ptr = 0; }

__attribute__((global))
void add_value(float* a, float* b, float* res) {
    *res = *a + *b;

    clean_value(a);
}
