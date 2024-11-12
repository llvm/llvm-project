// REQUIRES: comgr-has-spirv
// COM: Same as spirv-translator but with the cache

// COM: Generate a spirv-targeted LLVM IR file from an OpenCL kernel
// RUN: clang -c -emit-llvm --target=spirv64 %s -o %t.bc

// COM: Translate LLVM IR to SPIRV format
// RUN: amd-llvm-spirv --spirv-target-env=CL2.0 %t.bc -o %t.spv

// COM: Run Comgr Translator to covert SPIRV back to LLVM IR
// RUN: export AMD_COMGR_CACHE=1
// RUN: AMD_COMGR_CACHE_DIR=%t.cache spirv-translator %t.spv -o %t.translated.bc
// RUN: COUNT=$(ls "%t.cache" | wc -l)
// RUN: [ 2 -eq $COUNT ]

// COM: Dissasemble LLVM IR bitcode to LLVM IR text
// RUN: llvm-dis %t.translated.bc -o - | FileCheck %S/spirv-translator.cl
