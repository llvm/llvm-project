// REQUIRES: comgr-has-spirv
// COM: Enable this test once changes from amdspirv docker land

// COM: Generate a spirv-targeted LLVM IR file from an OpenCL kernel
// RUN: clang -c -emit-llvm --target=spirv64 %s -o %t.bc

// COM: Translate LLVM IR to SPIRV format
// RUN: amd-llvm-spirv %t.bc -o %t.spv

// COM: Run Comgr Translator to covert SPIRV back to LLVM IR
// RUN: spirv-translator %t.spv -o %t.translated.bc

// COM: Dissasemble LLVM IR bitcode to LLVM IR text
// RUN: llvm-dis %t.translated.bc -o - | FileCheck %s

// COM: Verify LLVM IR text
// CHECK: target triple = "spir64-unknown-unknown"
// CHECK: define spir_kernel void @source

void kernel source(__global int *j) {
  *j += 2;
}

