// REQUIRES: aarch64-registered-target

// RUN:                   %clang_cc1 -triple arm64-none-linux-gnu -target-feature +fullfp16 -disable-O0-optnone           -emit-llvm -o - %s | opt -S -passes=mem2reg             | FileCheck %s --check-prefixes=ALL,LLVM
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +fullfp16 -disable-O0-optnone -fclangir -emit-llvm -o - %s | opt -S -passes=mem2reg,simplifycfg | FileCheck %s --check-prefixes=ALL,LLVM %}
// RUN: %if cir-enabled %{%clang_cc1 -triple arm64-none-linux-gnu -target-feature +fullfp16 -disable-O0-optnone -fclangir -emit-cir  -o - %s |                                      FileCheck %s --check-prefixes=ALL,CIR %}

//=============================================================================
// NOTES
//
// Tests for unconstrained intrinsics that require the fullfp16 extension.
//
// This file contains tests that were originally located in
//  *  clang/test/CodeGen/AArch64/v8.2a-fp16-intrinsics.c
// The main difference is the use of RUN lines that enable ClangIR lowering;
// therefore only builtins currently supported by ClangIR are tested here.
// Once ClangIR support is complete, this file is intended to replace the
// original test file.
//
// These intrinsics expand to code containing multiple compound and declaration
// statements rather than just plain function calls, which leads to:
//  * "scopes" at the CIR level, and then
//  * redundant branches at the LLVM IR level.  
// The default lowering path never generates those redundant LLVM IR branches,
// hence for CIR we use `opt -passes=simplifycfg` to reduce the control flow
// and to make LLVM IR match for all paths.
//
// ACLE section headings based on v2025Q2 of the ACLE specification:
//  * https://arm-software.github.io/acle/neon_intrinsics/advsimd.html#bitwise-equal-to-zero
//
// TODO: Remove `-simplifycfg` once CIR lowering includes the relevant
//       optimizations to reduce the CFG.
//=============================================================================

#include <arm_fp16.h>

//===------------------------------------------------------===//
// 2.5.1.1.  Addition
//===------------------------------------------------------===//
// ALL-LABEL: @test_vaddh_f16(
float16_t test_vaddh_f16(float16_t a, float16_t b) {
// CIR: {{%.*}} = cir.add {{%.*}}, {{%.*}} : !cir.f16

// LLVM-SAME: half {{.*}} [[A:%.*]], half{{.*}} [[B:%.*]]) {{.*}} {
// LLVM:  [[ADD:%.*]] = fadd half [[A]], [[B]]
// LLVM:  ret half [[ADD]]
  return vaddh_f16(a, b);
}

//===------------------------------------------------------===//
// 2.5.10.1.  Subtraction
//===------------------------------------------------------===//
// ALL-LABEL: @test_vsubh_f16(
float16_t test_vsubh_f16(float16_t a, float16_t b) {
// CIR: {{%.*}} = cir.sub {{%.*}}, {{%.*}} : !cir.f16

// LLVM-SAME: half {{.*}} [[A:%.]], half {{.*}} [[B:%.]]) {{.*}} {
// LLVM:  [[SUB:%.*]] = fsub half [[A]], [[B]]
// LLVM:  ret half [[SUB]]
  return vsubh_f16(a, b);
}

//===------------------------------------------------------===//
// 2.5.9.1.  Multiplication
//===------------------------------------------------------===//
// ALL-LABEL: @test_vmulh_f16(
float16_t test_vmulh_f16(float16_t a, float16_t b) {
// CIR: {{%.*}} = cir.mul {{%.*}}, {{%.*}} : !cir.f16

// LLVM-SAME: half {{.*}} [[A:%.]], half {{.*}} [[B:%.]]) {{.*}} {
// LLVM:  [[MUL:%.*]] = fmul half [[A]], [[B]]
// LLVM:  ret half [[MUL]]
  return vmulh_f16(a, b);
}

//===------------------------------------------------------===//
// 2.5.1.6.  Division
//===------------------------------------------------------===//
// ALL-LABEL: @test_vdivh_f16(
float16_t test_vdivh_f16(float16_t a, float16_t b) {
// CIR: {{%.*}} = cir.div {{%.*}}, {{%.*}} : !cir.f16

// LLVM-SAME: half {{.*}} [[A:%.]], half {{.*}} [[B:%.]]) {{.*}} {
// LLVM:  [[DIV:%.*]] = fdiv half [[A]], [[B]]
// LLVM:  ret half [[DIV]]
  return vdivh_f16(a, b);
}

//===------------------------------------------------------===//
// 2.5.2.1.  Bitwise equal to zero
//===------------------------------------------------------===//
// ALL-LABEL: test_vceqzh_f16
uint16_t test_vceqzh_f16(float16_t a) {
// CIR:   [[C_0:%.*]] = cir.const #cir.fp<0.000000e+00>
// CIR:   [[CMP:%.*]] = cir.cmp eq %{{.*}}, [[C_0]] : !cir.f16
// CIR:   [[RES:%.*]] = cir.cast bool_to_int [[CMP]] : !cir.bool -> !cir.int<s, 1>
// CIR:   cir.cast integral [[RES]] : !cir.int<s, 1> -> !u16i

// LLVM-SAME: (half {{.*}} [[A:%.*]])
// LLVM:  [[TMP1:%.*]] = fcmp oeq half [[A]], 0xH0000
// LLVM:  [[TMP2:%.*]] = sext i1 [[TMP1]] to i16
// LLVM:  ret i16 [[TMP2]]
  return vceqzh_f16(a);
}

//===------------------------------------------------------===//
// 2.5.1.1.1. Absolute value
//===------------------------------------------------------===//
// ALL-LABEL: @test_vabsh_f16
float16_t test_vabsh_f16(float16_t a) {
// CIR: {{%.*}} = cir.fabs {{%.*}} : !cir.f16

// LLVM-SAME: (half{{.*}} [[A:%.*]])
// LLVM:  [[ABS:%.*]] = call half @llvm.fabs.f16(half [[A]])
// LLVM:  ret half [[ABS]]
  return vabsh_f16(a);
}

//===------------------------------------------------------===//
// 2.5.1.1.2. Absolute difference
//===------------------------------------------------------===//
// ALL-LABEL: test_vabdh_f16
float16_t test_vabdh_f16(float16_t a, float16_t b) {
// CIR:  cir.call_llvm_intrinsic "aarch64.sisd.fabd" {{.*}} -> !cir.f16

// LLVM-SAME: half {{.*}} [[A:%.*]], half {{.*}} [[B:%.*]])
// LLVM:  [[ABD:%.*]] = call half @llvm.aarch64.sisd.fabd.f16(half [[A]], half [[B]])
// LLVM:  ret half [[ABD]]
  return vabdh_f16(a, b);
}

//===------------------------------------------------------===//
// 2.5.1.2.1.  Reciprocal estimate
//===------------------------------------------------------===//
// ALL-LABEL: test_vrecpeh_f16
float16_t test_vrecpeh_f16(float16_t a) {
// CIR:  cir.call_llvm_intrinsic "aarch64.neon.frecpe" {{.*}} -> !cir.f16

// LLVM-SAME: half {{.*}} [[A:%.*]])
// LLVM: [[VREC:%.*]] = call half @llvm.aarch64.neon.frecpe.f16(half [[A]])
// LLVM: ret half [[VREC]]
  return vrecpeh_f16(a);
}

// ALL-LABEL: test_vrecpxh_f16
float16_t test_vrecpxh_f16(float16_t a) {
// CIR:  cir.call_llvm_intrinsic "aarch64.neon.frecpx" {{.*}} -> !cir.f16

// LLVM-SAME: half {{.*}} [[A:%.*]])
// LLVM: [[VREC:%.*]] = call half @llvm.aarch64.neon.frecpx.f16(half [[A]])
// LLVM: ret half [[VREC]]
  return vrecpxh_f16(a);
}

//===------------------------------------------------------===//
// 2.5.1.2.2.  Reciprocal square-root estimate
//===------------------------------------------------------===//
// ALL-LABEL: test_vrsqrteh_f16
float16_t test_vrsqrteh_f16(float16_t a) {
// CIR:  cir.call_llvm_intrinsic "aarch64.neon.frsqrte" {{.*}} -> !cir.f16

// LLVM-SAME: half {{.*}} [[A:%.*]])
// LLVM:  [[RND:%.*]] = call half @llvm.aarch64.neon.frsqrte.f16(half [[A]])
// LLVM:  ret half [[RND]]
  return vrsqrteh_f16(a);
}

// ALL-LABEL: test_vrsqrtsh_f16
float16_t test_vrsqrtsh_f16(float16_t a, float16_t b) {
// CIR:  cir.call_llvm_intrinsic "aarch64.neon.frsqrts" {{.*}} -> !cir.f16

// LLVM-SAME: half {{.*}} [[A:%.*]], half {{.*}} [[B:%.*]])
// LLVM:  [[RSQRTS:%.*]] = call half @llvm.aarch64.neon.frsqrts.f16(half [[A]], half [[B]])
// LLVM:  ret half [[RSQRTS]]
  return vrsqrtsh_f16(a, b);
}

//===------------------------------------------------------===//
// 2.5.4.1. Negate
//===------------------------------------------------------===//
// ALL-LABEL: @test_vnegh_f16
float16_t test_vnegh_f16(float16_t a) {
// CIR: cir.minus {{.*}} : !cir.f16

// LLVM-SAME: half{{.*}} [[A:%.*]])
// LLVM: [[NEG:%.*]] = fneg half [[A:%.*]]
// LLVM: ret half [[NEG]]
  return vnegh_f16(a);
}

//===------------------------------------------------------===//
// 2.5.1.9.3 Fused multiply-accumulate
//===------------------------------------------------------===//
// ALL-LABEL: test_vfmah_f16
float16_t test_vfmah_f16(float16_t a, float16_t b, float16_t c) {
// CIR: cir.call_llvm_intrinsic "fma" {{.*}} : (!cir.f16, !cir.f16, !cir.f16) -> !cir.f16

// LLVM-SAME: half{{.*}} [[A:%.*]], half{{.*}} [[B:%.*]], half{{.*}} [[C:%.*]])
// LLVM:  [[FMA:%.*]] = call half @llvm.fma.f16(half [[B]], half [[C]], half [[A]])
// LLVM:  ret half [[FMA]]
  return vfmah_f16(a, b, c);
}

// ALL-LABEL: test_vfmsh_f16
float16_t test_vfmsh_f16(float16_t a, float16_t b, float16_t c) {
// CIR: [[SUB:%.*]] = cir.minus %{{.*}} : !cir.f16
// CIR: cir.call_llvm_intrinsic "fma" [[SUB]], {{.*}} : (!cir.f16, !cir.f16, !cir.f16) -> !cir.f16

// LLVM-SAME: half{{.*}} [[A:%.*]], half{{.*}} [[B:%.*]], half{{.*}} [[C:%.*]])
// LLVM:  [[SUB:%.*]] = fneg half [[B]]
// LLVM:  [[ADD:%.*]] = call half @llvm.fma.f16(half [[SUB]], half [[C]], half [[A]])
// LLVM:  ret half [[ADD]]
  return vfmsh_f16(a, b, c);
}
