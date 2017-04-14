// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx900 -S -emit-llvm -o - %s | FileCheck --check-prefix=DEFAULT %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx900 -S -emit-llvm -o - -target-feature +fp32-denormals %s | FileCheck --check-prefix=FEATURE_FP32_DENORMALS_ON %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx900 -S -emit-llvm -o - -target-feature -fp32-denormals %s | FileCheck --check-prefix=FEATURE_FP32_DENORMALS_OFF %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx900 -S -emit-llvm -o - -cl-denorms-are-zero %s | FileCheck --check-prefix=OPT_DENORMS_ARE_ZERO %s

// DEFAULT: +fp32-denormals
// FEATURE_FP32_DENORMALS_ON: +fp32-denormals
// FEATURE_FP32_DENORMALS_OFF: -fp32-denormals
// OPT_DENORMS_ARE_ZERO: -fp32-denormals

kernel void gfx9_fp32_denorms() {}
