// REQUIRES: amdgpu-registered-target

// Check that appropriate features are defined for every supported AMDGPU
// "-target" and "-mcpu" options.

// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx904 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX904 %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx906 -S -emit-llvm -o - %s | FileCheck --check-prefix=GFX906 %s

// GFX904: "target-features"="+16-bit-insts,+dpp,+fp32-denormals,+fp64-fp16-denormals,+gfx9-insts,+s-memrealtime"
// GFX906: "target-features"="+16-bit-insts,+dl-insts,+dpp,+fp32-denormals,+fp64-fp16-denormals,+gfx9-insts,+s-memrealtime"

kernel void test() {}
