// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +fullfp16 \
// RUN: -S -disable-O0-optnone \
// RUN: -emit-llvm -o - %s | opt -S -passes=mem2reg \
// RUN: | FileCheck --check-prefixes=COMMON,COMMONIR,UNCONSTRAINED %s
// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +fullfp16 \
// RUN: -S -disable-O0-optnone \
// RUN: -ffp-exception-behavior=strict -fexperimental-strict-floating-point -emit-llvm -o - %s | opt -S -passes=mem2reg \
// RUN: | FileCheck --check-prefixes=COMMON,COMMONIR,CONSTRAINED %s

// REQUIRES: aarch64-registered-target

#include <arm_fp16.h>

// COMMON-LABEL: test_vceqzh_f16
// UNCONSTRAINED:  [[TMP1:%.*]] = fcmp oeq half %a, 0xH0000
// CONSTRAINED:    [[TMP1:%.*]] = call i1 @llvm.experimental.constrained.fcmp.f16(half %a, half 0xH0000, metadata !"oeq", metadata !"fpexcept.strict")
// COMMONIR:       [[TMP2:%.*]] = sext i1 [[TMP1]] to i16
// COMMONIR:       ret i16 [[TMP2]]
uint16_t test_vceqzh_f16(float16_t a) {
  return vceqzh_f16(a);
}

// COMMON-LABEL: test_vcgezh_f16
// UNCONSTRAINED:  [[TMP1:%.*]] = fcmp oge half %a, 0xH0000
// CONSTRAINED:    [[TMP1:%.*]] = call i1 @llvm.experimental.constrained.fcmps.f16(half %a, half 0xH0000, metadata !"oge", metadata !"fpexcept.strict")
// COMMONIR:       [[TMP2:%.*]] = sext i1 [[TMP1]] to i16
// COMMONIR:       ret i16 [[TMP2]]
uint16_t test_vcgezh_f16(float16_t a) {
  return vcgezh_f16(a);
}

// COMMON-LABEL: test_vcgtzh_f16
// UNCONSTRAINED:  [[TMP1:%.*]] = fcmp ogt half %a, 0xH0000
// CONSTRAINED:    [[TMP1:%.*]] = call i1 @llvm.experimental.constrained.fcmps.f16(half %a, half 0xH0000, metadata !"ogt", metadata !"fpexcept.strict")
// COMMONIR:       [[TMP2:%.*]] = sext i1 [[TMP1]] to i16
// COMMONIR:       ret i16 [[TMP2]]
uint16_t test_vcgtzh_f16(float16_t a) {
  return vcgtzh_f16(a);
}

// COMMON-LABEL: test_vclezh_f16
// UNCONSTRAINED:  [[TMP1:%.*]] = fcmp ole half %a, 0xH0000
// CONSTRAINED:    [[TMP1:%.*]] = call i1 @llvm.experimental.constrained.fcmps.f16(half %a, half 0xH0000, metadata !"ole", metadata !"fpexcept.strict")
// COMMONIR:       [[TMP2:%.*]] = sext i1 [[TMP1]] to i16
// COMMONIR:       ret i16 [[TMP2]]
uint16_t test_vclezh_f16(float16_t a) {
  return vclezh_f16(a);
}

// COMMON-LABEL: test_vcltzh_f16
// UNCONSTRAINED:  [[TMP1:%.*]] = fcmp olt half %a, 0xH0000
// CONSTRAINED:    [[TMP1:%.*]] = call i1 @llvm.experimental.constrained.fcmps.f16(half %a, half 0xH0000, metadata !"olt", metadata !"fpexcept.strict")
// COMMONIR:       [[TMP2:%.*]] = sext i1 [[TMP1]] to i16
// COMMONIR:       ret i16 [[TMP2]]
uint16_t test_vcltzh_f16(float16_t a) {
  return vcltzh_f16(a);
}

// COMMON-LABEL: test_vcvth_f16_s16
// UNCONSTRAINED:  [[VCVT:%.*]] = sitofp i16 %a to half
// CONSTRAINED:    [[VCVT:%.*]] = call half @llvm.experimental.constrained.sitofp.f16.i16(i16 %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// COMMONIR:       ret half [[VCVT]]
float16_t test_vcvth_f16_s16 (int16_t a) {
  return vcvth_f16_s16(a);
}

// COMMON-LABEL: test_vcvth_f16_s32
// UNCONSTRAINED:  [[VCVT:%.*]] = sitofp i32 %a to half
// CONSTRAINED:    [[VCVT:%.*]] = call half @llvm.experimental.constrained.sitofp.f16.i32(i32 %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// COMMONIR:       ret half [[VCVT]]
float16_t test_vcvth_f16_s32 (int32_t a) {
  return vcvth_f16_s32(a);
}

// COMMON-LABEL: test_vcvth_f16_s64
// UNCONSTRAINED:  [[VCVT:%.*]] = sitofp i64 %a to half
// CONSTRAINED:    [[VCVT:%.*]] = call half @llvm.experimental.constrained.sitofp.f16.i64(i64 %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// COMMONIR:       ret half [[VCVT]]
float16_t test_vcvth_f16_s64 (int64_t a) {
  return vcvth_f16_s64(a);
}

// COMMON-LABEL: test_vcvth_f16_u16
// UNCONSTRAINED:  [[VCVT:%.*]] = uitofp i16 %a to half
// CONSTRAINED:  [[VCVT:%.*]] = call half @llvm.experimental.constrained.uitofp.f16.i16(i16 %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// COMMONIR:       ret half [[VCVT]]
float16_t test_vcvth_f16_u16 (uint16_t a) {
  return vcvth_f16_u16(a);
}

// COMMON-LABEL: test_vcvth_f16_u32
// UNCONSTRAINED:  [[VCVT:%.*]] = uitofp i32 %a to half
// CONSTRAINED:    [[VCVT:%.*]] = call half @llvm.experimental.constrained.uitofp.f16.i32(i32 %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// COMMONIR:  ret half [[VCVT]]
float16_t test_vcvth_f16_u32 (uint32_t a) {
  return vcvth_f16_u32(a);
}

// COMMON-LABEL: test_vcvth_f16_u64
// UNCONSTRAINED:  [[VCVT:%.*]] = uitofp i64 %a to half
// CONSTRAINED:    [[VCVT:%.*]] = call half @llvm.experimental.constrained.uitofp.f16.i64(i64 %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// COMMONIR:       ret half [[VCVT]]
float16_t test_vcvth_f16_u64 (uint64_t a) {
  return vcvth_f16_u64(a);
}

// COMMON-LABEL: test_vcvth_s16_f16
// COMMONIR:       [[VCVT:%.*]] = call i32 @llvm.aarch64.neon.fcvtzs.i32.f16(half %a)
// COMMONIR:       [[TRUNC:%.*]] = trunc i32 [[VCVT]] to i16
// COMMONIR:       ret i16 [[TRUNC]]
int16_t test_vcvth_s16_f16 (float16_t a) {
  return vcvth_s16_f16(a);
}

// COMMON-LABEL: test_vcvth_s32_f16
// COMMONIR:       [[VCVT:%.*]] = call i32 @llvm.aarch64.neon.fcvtzs.i32.f16(half %a)
// COMMONIR:       ret i32 [[VCVT]]
int32_t test_vcvth_s32_f16 (float16_t a) {
  return vcvth_s32_f16(a);
}

// COMMON-LABEL: test_vcvth_s64_f16
// COMMONIR:       [[VCVT:%.*]] = call i64 @llvm.aarch64.neon.fcvtzs.i64.f16(half %a)
// COMMONIR:       ret i64 [[VCVT]]
int64_t test_vcvth_s64_f16 (float16_t a) {
  return vcvth_s64_f16(a);
}

// COMMON-LABEL: test_vcvth_u16_f16
// COMMONIR:       [[VCVT:%.*]] = call i32 @llvm.aarch64.neon.fcvtzu.i32.f16(half %a)
// COMMONIR:       [[TRUNC:%.*]] = trunc i32 [[VCVT]] to i16
// COMMONIR:       ret i16 [[TRUNC]]
uint16_t test_vcvth_u16_f16 (float16_t a) {
  return vcvth_u16_f16(a);
}

// COMMON-LABEL: test_vcvth_u32_f16
// COMMONIR:       [[VCVT:%.*]] = call i32 @llvm.aarch64.neon.fcvtzu.i32.f16(half %a)
// COMMONIR:       ret i32 [[VCVT]]
uint32_t test_vcvth_u32_f16 (float16_t a) {
  return vcvth_u32_f16(a);
}

// COMMON-LABEL: test_vcvth_u64_f16
// COMMONIR:       [[VCVT:%.*]] = call i64 @llvm.aarch64.neon.fcvtzu.i64.f16(half %a)
// COMMONIR:       ret i64 [[VCVT]]
uint64_t test_vcvth_u64_f16 (float16_t a) {
  return vcvth_u64_f16(a);
}

// COMMON-LABEL: test_vrndh_f16
// UNCONSTRAINED:  [[RND:%.*]] = call half @llvm.trunc.f16(half %a)
// CONSTRAINED:    [[RND:%.*]] = call half @llvm.experimental.constrained.trunc.f16(half %a, metadata !"fpexcept.strict")
// COMMONIR:       ret half [[RND]]
float16_t test_vrndh_f16(float16_t a) {
  return vrndh_f16(a);
}

// COMMON-LABEL: test_vrndah_f16
// UNCONSTRAINED:  [[RND:%.*]] = call half @llvm.round.f16(half %a)
// CONSTRAINED:    [[RND:%.*]] = call half @llvm.experimental.constrained.round.f16(half %a, metadata !"fpexcept.strict")
// COMMONIR:       ret half [[RND]]
float16_t test_vrndah_f16(float16_t a) {
  return vrndah_f16(a);
}

// COMMON-LABEL: test_vrndih_f16
// UNCONSTRAINED:  [[RND:%.*]] = call half @llvm.nearbyint.f16(half %a)
// CONSTRAINED:    [[RND:%.*]] = call half @llvm.experimental.constrained.nearbyint.f16(half %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// COMMONIR:       ret half [[RND]]
float16_t test_vrndih_f16(float16_t a) {
  return vrndih_f16(a);
}

// COMMON-LABEL: test_vrndmh_f16
// UNCONSTRAINED:  [[RND:%.*]] = call half @llvm.floor.f16(half %a)
// CONSTRAINED:    [[RND:%.*]] = call half @llvm.experimental.constrained.floor.f16(half %a, metadata !"fpexcept.strict")
// COMMONIR:       ret half [[RND]]
float16_t test_vrndmh_f16(float16_t a) {
  return vrndmh_f16(a);
}

// COMMON-LABEL: test_vrndph_f16
// UNCONSTRAINED:  [[RND:%.*]] = call half @llvm.ceil.f16(half %a)
// CONSTRAINED:    [[RND:%.*]] = call half @llvm.experimental.constrained.ceil.f16(half %a, metadata !"fpexcept.strict")
// COMMONIR:       ret half [[RND]]
float16_t test_vrndph_f16(float16_t a) {
  return vrndph_f16(a);
}

// COMMON-LABEL: test_vrndxh_f16
// UNCONSTRAINED:  [[RND:%.*]] = call half @llvm.rint.f16(half %a)
// CONSTRAINED:    [[RND:%.*]] = call half @llvm.experimental.constrained.rint.f16(half %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// COMMONIR:       ret half [[RND]]
float16_t test_vrndxh_f16(float16_t a) {
  return vrndxh_f16(a);
}

// COMMON-LABEL: test_vsqrth_f16
// UNCONSTRAINED:  [[SQR:%.*]] = call half @llvm.sqrt.f16(half %a)
// CONSTRAINED:    [[SQR:%.*]] = call half @llvm.experimental.constrained.sqrt.f16(half %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// COMMONIR:       ret half [[SQR]]
float16_t test_vsqrth_f16(float16_t a) {
  return vsqrth_f16(a);
}

// COMMON-LABEL: test_vaddh_f16
// UNCONSTRAINED:  [[ADD:%.*]] = fadd half %a, %b
// CONSTRAINED:    [[ADD:%.*]] = call half @llvm.experimental.constrained.fadd.f16(half %a, half %b, metadata !"round.tonearest", metadata !"fpexcept.strict")
// COMMONIR:       ret half [[ADD]]
float16_t test_vaddh_f16(float16_t a, float16_t b) {
  return vaddh_f16(a, b);
}

// COMMON-LABEL: test_vceqh_f16
// UNCONSTRAINED:  [[TMP1:%.*]] = fcmp oeq half %a, %b
// CONSTRAINED:    [[TMP1:%.*]] = call i1 @llvm.experimental.constrained.fcmp.f16(half %a, half %b, metadata !"oeq", metadata !"fpexcept.strict")
// COMMONIR:       [[TMP2:%.*]] = sext i1 [[TMP1]] to i16
// COMMONIR:       ret i16 [[TMP2]]
uint16_t test_vceqh_f16(float16_t a, float16_t b) {
  return vceqh_f16(a, b);
}

// COMMON-LABEL: test_vcgeh_f16
// UNCONSTRAINED:  [[TMP1:%.*]] = fcmp oge half %a, %b
// CONSTRAINED:    [[TMP1:%.*]] = call i1 @llvm.experimental.constrained.fcmps.f16(half %a, half %b, metadata !"oge", metadata !"fpexcept.strict")
// COMMONIR:       [[TMP2:%.*]] = sext i1 [[TMP1]] to i16
// COMMONIR:       ret i16 [[TMP2]]
uint16_t test_vcgeh_f16(float16_t a, float16_t b) {
  return vcgeh_f16(a, b);
}

// COMMON-LABEL: test_vcgth_f16
// UNCONSTRAINED:  [[TMP1:%.*]] = fcmp ogt half %a, %b
// CONSTRAINED:    [[TMP1:%.*]] = call i1 @llvm.experimental.constrained.fcmps.f16(half %a, half %b, metadata !"ogt", metadata !"fpexcept.strict")
// COMMONIR:       [[TMP2:%.*]] = sext i1 [[TMP1]] to i16
// COMMONIR:       ret i16 [[TMP2]]
uint16_t test_vcgth_f16(float16_t a, float16_t b) {
  return vcgth_f16(a, b);
}

// COMMON-LABEL: test_vcleh_f16
// UNCONSTRAINED:  [[TMP1:%.*]] = fcmp ole half %a, %b
// CONSTRAINED:    [[TMP1:%.*]] = call i1 @llvm.experimental.constrained.fcmps.f16(half %a, half %b, metadata !"ole", metadata !"fpexcept.strict")
// COMMONIR:       [[TMP2:%.*]] = sext i1 [[TMP1]] to i16
// COMMONIR:       ret i16 [[TMP2]]
uint16_t test_vcleh_f16(float16_t a, float16_t b) {
  return vcleh_f16(a, b);
}

// COMMON-LABEL: test_vclth_f16
// UNCONSTRAINED:  [[TMP1:%.*]] = fcmp olt half %a, %b
// CONSTRAINED:    [[TMP1:%.*]] = call i1 @llvm.experimental.constrained.fcmps.f16(half %a, half %b, metadata !"olt", metadata !"fpexcept.strict")
// COMMONIR:       [[TMP2:%.*]] = sext i1 [[TMP1]] to i16
// COMMONIR:       ret i16 [[TMP2]]
uint16_t test_vclth_f16(float16_t a, float16_t b) {
  return vclth_f16(a, b);
}

// COMMON-LABEL: test_vdivh_f16
// UNCONSTRAINED:  [[DIV:%.*]] = fdiv half %a, %b
// CONSTRAINED:    [[DIV:%.*]] = call half @llvm.experimental.constrained.fdiv.f16(half %a, half %b, metadata !"round.tonearest", metadata !"fpexcept.strict")
// COMMONIR:       ret half [[DIV]]
float16_t test_vdivh_f16(float16_t a, float16_t b) {
  return vdivh_f16(a, b);
}

// COMMON-LABEL: test_vmulh_f16
// UNCONSTRAINED:  [[MUL:%.*]] = fmul half %a, %b
// CONSTRAINED:  [[MUL:%.*]] = call half @llvm.experimental.constrained.fmul.f16(half %a, half %b, metadata !"round.tonearest", metadata !"fpexcept.strict")
// COMMONIR:       ret half [[MUL]]
float16_t test_vmulh_f16(float16_t a, float16_t b) {
  return vmulh_f16(a, b);
}

// COMMON-LABEL: test_vsubh_f16
// UNCONSTRAINED:  [[SUB:%.*]] = fsub half %a, %b
// CONSTRAINED:    [[SUB:%.*]] = call half @llvm.experimental.constrained.fsub.f16(half %a, half %b, metadata !"round.tonearest", metadata !"fpexcept.strict")
// COMMONIR:       ret half [[SUB]]
float16_t test_vsubh_f16(float16_t a, float16_t b) {
  return vsubh_f16(a, b);
}

// COMMON-LABEL: test_vfmah_f16
// UNCONSTRAINED:  [[FMA:%.*]] = call half @llvm.fma.f16(half %b, half %c, half %a)
// CONSTRAINED:    [[FMA:%.*]] = call half @llvm.experimental.constrained.fma.f16(half %b, half %c, half %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// COMMONIR:       ret half [[FMA]]
float16_t test_vfmah_f16(float16_t a, float16_t b, float16_t c) {
  return vfmah_f16(a, b, c);
}

// COMMON-LABEL: test_vfmsh_f16
// UNCONSTRAINED:  [[SUB:%.*]] = fsub half 0xH8000, %b
// CONSTRAINED:    [[SUB:%.*]] = call half @llvm.experimental.constrained.fsub.f16(half 0xH8000, half %b, metadata !"round.tonearest", metadata !"fpexcept.strict")
// UNCONSTRAINED:  [[ADD:%.*]] = call half @llvm.fma.f16(half [[SUB]], half %c, half %a)
// CONSTRAINED:    [[ADD:%.*]] = call half @llvm.experimental.constrained.fma.f16(half [[SUB]], half %c, half %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// COMMONIR:       ret half [[ADD]]
float16_t test_vfmsh_f16(float16_t a, float16_t b, float16_t c) {
  return vfmsh_f16(a, b, c);
}

