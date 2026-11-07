// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64 -target-feature +sve -emit-llvm -disable-O0-optnone -mreassociate -o - %s | opt -S -passes=mem2reg | FileCheck %s --check-prefixes=CHECK,FENCE
// RUN: %clang_cc1 -triple aarch64 -target-feature +sve -emit-llvm -disable-O0-optnone -o - %s | opt -S -passes=mem2reg | FileCheck %s --check-prefixes=CHECK,IDENTITY --implicit-check-not=llvm.arithmetic.fence
// RUN: %clang_cc1 -triple aarch64 -target-feature +sve -O2 -mreassociate -ffp-contract=fast -S -o - %s | FileCheck %s --check-prefix=ASM

#include <arm_sve.h>

// CHECK-LABEL: @test_f16(
// FENCE: [[F16:%.*]] = call reassoc <vscale x 8 x half> @llvm.arithmetic.fence.nxv8f16(<vscale x 8 x half> %x)
// FENCE-NEXT: ret <vscale x 8 x half> [[F16]]
// IDENTITY: ret <vscale x 8 x half> %x
svfloat16_t test_f16(svfloat16_t x) { return __arithmetic_fence(x); }

// CHECK-LABEL: @test_f32(
// FENCE: [[F32:%.*]] = call reassoc <vscale x 4 x float> @llvm.arithmetic.fence.nxv4f32(<vscale x 4 x float> %x)
// FENCE-NEXT: ret <vscale x 4 x float> [[F32]]
// IDENTITY: ret <vscale x 4 x float> %x
svfloat32_t test_f32(svfloat32_t x) { return __arithmetic_fence(x); }

// CHECK-LABEL: @test_f64(
// FENCE: [[F64:%.*]] = call reassoc <vscale x 2 x double> @llvm.arithmetic.fence.nxv2f64(<vscale x 2 x double> %x)
// FENCE-NEXT: ret <vscale x 2 x double> [[F64]]
// IDENTITY: ret <vscale x 2 x double> %x
svfloat64_t test_f64(svfloat64_t x) { return __arithmetic_fence(x); }

// CHECK-LABEL: @test_bf16(
// FENCE: [[BF16:%.*]] = call reassoc <vscale x 8 x bfloat> @llvm.arithmetic.fence.nxv8bf16(<vscale x 8 x bfloat> %x)
// FENCE-NEXT: ret <vscale x 8 x bfloat> [[BF16]]
// IDENTITY: ret <vscale x 8 x bfloat> %x
svbfloat16_t test_bf16(svbfloat16_t x) { return __arithmetic_fence(x); }

// ASM-LABEL: test_fenced_muladd:
// ASM: fmul z
// ASM: fadd z
// ASM: ret
svfloat32_t test_fenced_muladd(svfloat32_t a, svfloat32_t b, svfloat32_t c) {
  return __arithmetic_fence(a * b) + c;
}

// ASM-LABEL: test_unfenced_muladd:
// ASM: {{fmad|fmla}} z
// ASM: ret
svfloat32_t test_unfenced_muladd(svfloat32_t a, svfloat32_t b, svfloat32_t c) {
  return a * b + c;
}
