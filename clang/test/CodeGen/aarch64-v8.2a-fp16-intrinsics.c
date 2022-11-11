// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +fullfp16\
// RUN: -S -disable-O0-optnone -emit-llvm -o - %s \
// RUN: | opt -S -passes=mem2reg \
// RUN: | FileCheck %s

// REQUIRES: aarch64-registered-target

#include <arm_fp16.h>

// CHECK-LABEL: test_vabsh_f16
// CHECK:  [[ABS:%.*]] =  call half @llvm.fabs.f16(half %a)
// CHECK:  ret half [[ABS]]
float16_t test_vabsh_f16(float16_t a) {
  return vabsh_f16(a);
}

// CHECK-LABEL: test_vceqzh_f16
// CHECK:  [[TMP1:%.*]] = fcmp oeq half %a, 0xH0000
// CHECK:  [[TMP2:%.*]] = sext i1 [[TMP1]] to i16
// CHECK:  ret i16 [[TMP2]]
uint16_t test_vceqzh_f16(float16_t a) {
  return vceqzh_f16(a);
}

// CHECK-LABEL: test_vcgezh_f16
// CHECK:  [[TMP1:%.*]] = fcmp oge half %a, 0xH0000
// CHECK:  [[TMP2:%.*]] = sext i1 [[TMP1]] to i16
// CHECK:  ret i16 [[TMP2]]
uint16_t test_vcgezh_f16(float16_t a) {
  return vcgezh_f16(a);
}

// CHECK-LABEL: test_vcgtzh_f16
// CHECK:  [[TMP1:%.*]] = fcmp ogt half %a, 0xH0000
// CHECK:  [[TMP2:%.*]] = sext i1 [[TMP1]] to i16
// CHECK:  ret i16 [[TMP2]]
uint16_t test_vcgtzh_f16(float16_t a) {
  return vcgtzh_f16(a);
}

// CHECK-LABEL: test_vclezh_f16
// CHECK:  [[TMP1:%.*]] = fcmp ole half %a, 0xH0000
// CHECK:  [[TMP2:%.*]] = sext i1 [[TMP1]] to i16
// CHECK:  ret i16 [[TMP2]]
uint16_t test_vclezh_f16(float16_t a) {
  return vclezh_f16(a);
}

// CHECK-LABEL: test_vcltzh_f16
// CHECK:  [[TMP1:%.*]] = fcmp olt half %a, 0xH0000
// CHECK:  [[TMP2:%.*]] = sext i1 [[TMP1]] to i16
// CHECK:  ret i16 [[TMP2]]
uint16_t test_vcltzh_f16(float16_t a) {
  return vcltzh_f16(a);
}

// CHECK-LABEL: test_vcvth_f16_s16
// CHECK:  [[VCVT:%.*]] = sitofp i16 %a to half
// CHECK:  ret half [[VCVT]]
float16_t test_vcvth_f16_s16 (int16_t a) {
  return vcvth_f16_s16(a);
}

// CHECK-LABEL: test_vcvth_f16_s32
// CHECK:  [[VCVT:%.*]] = sitofp i32 %a to half
// CHECK:  ret half [[VCVT]]
float16_t test_vcvth_f16_s32 (int32_t a) {
  return vcvth_f16_s32(a);
}

// CHECK-LABEL: test_vcvth_f16_s64
// CHECK:  [[VCVT:%.*]] = sitofp i64 %a to half
// CHECK:  ret half [[VCVT]]
float16_t test_vcvth_f16_s64 (int64_t a) {
  return vcvth_f16_s64(a);
}

// CHECK-LABEL: test_vcvth_f16_u16
// CHECK:  [[VCVT:%.*]] = uitofp i16 %a to half
// CHECK:  ret half [[VCVT]]
float16_t test_vcvth_f16_u16 (uint16_t a) {
  return vcvth_f16_u16(a);
}

// CHECK-LABEL: test_vcvth_f16_u32
// CHECK:  [[VCVT:%.*]] = uitofp i32 %a to half
// CHECK:  ret half [[VCVT]]
float16_t test_vcvth_f16_u32 (uint32_t a) {
  return vcvth_f16_u32(a);
}

// CHECK-LABEL: test_vcvth_f16_u64
// CHECK:  [[VCVT:%.*]] = uitofp i64 %a to half
// CHECK:  ret half [[VCVT]]
float16_t test_vcvth_f16_u64 (uint64_t a) {
  return vcvth_f16_u64(a);
}

// CHECK-LABEL: test_vcvth_s16_f16
// CHECK:  [[VCVT:%.*]] = call i32 @llvm.aarch64.neon.fcvtzs.i32.f16(half %a)
// CHECK:  [[TRUNC:%.*]] = trunc i32 [[VCVT]] to i16
// CHECK:  ret i16 [[TRUNC]]
int16_t test_vcvth_s16_f16 (float16_t a) {
  return vcvth_s16_f16(a);
}

// CHECK-LABEL: test_vcvth_s32_f16
// CHECK:  [[VCVT:%.*]] = call i32 @llvm.aarch64.neon.fcvtzs.i32.f16(half %a)
// CHECK:  ret i32 [[VCVT]]
int32_t test_vcvth_s32_f16 (float16_t a) {
  return vcvth_s32_f16(a);
}

// CHECK-LABEL: test_vcvth_s64_f16
// CHECK:  [[VCVT:%.*]] = call i64 @llvm.aarch64.neon.fcvtzs.i64.f16(half %a)
// CHECK:  ret i64 [[VCVT]]
int64_t test_vcvth_s64_f16 (float16_t a) {
  return vcvth_s64_f16(a);
}

// CHECK-LABEL: test_vcvth_u16_f16
// CHECK:  [[VCVT:%.*]] = call i32 @llvm.aarch64.neon.fcvtzu.i32.f16(half %a)
// CHECK:  [[TRUNC:%.*]] = trunc i32 [[VCVT]] to i16
// CHECK:  ret i16 [[TRUNC]]
uint16_t test_vcvth_u16_f16 (float16_t a) {
  return vcvth_u16_f16(a);
}

// CHECK-LABEL: test_vcvth_u32_f16
// CHECK:  [[VCVT:%.*]] = call i32 @llvm.aarch64.neon.fcvtzu.i32.f16(half %a)
// CHECK:  ret i32 [[VCVT]]
uint32_t test_vcvth_u32_f16 (float16_t a) {
  return vcvth_u32_f16(a);
}

// CHECK-LABEL: test_vcvth_u64_f16
// CHECK:  [[VCVT:%.*]] = call i64 @llvm.aarch64.neon.fcvtzu.i64.f16(half %a)
// CHECK:  ret i64 [[VCVT]]
uint64_t test_vcvth_u64_f16 (float16_t a) {
  return vcvth_u64_f16(a);
}

// CHECK-LABEL: test_vcvtah_s16_f16
// CHECK: [[FCVT:%.*]] = call i32 @llvm.aarch64.neon.fcvtas.i32.f16(half %a)
// CHECK: [[RET:%.*]] = trunc i32 [[FCVT]] to i16
// CHECK: ret i16 [[RET]]
int16_t test_vcvtah_s16_f16 (float16_t a) {
  return vcvtah_s16_f16(a);
}

// CHECK-LABEL: test_vcvtah_s32_f16
// CHECK: [[VCVT:%.*]] = call i32 @llvm.aarch64.neon.fcvtas.i32.f16(half %a)
// CHECK: ret i32 [[VCVT]]
int32_t test_vcvtah_s32_f16 (float16_t a) {
  return vcvtah_s32_f16(a);
}

// CHECK-LABEL: test_vcvtah_s64_f16
// CHECK: [[VCVT:%.*]] = call i64 @llvm.aarch64.neon.fcvtas.i64.f16(half %a)
// CHECK: ret i64 [[VCVT]]
int64_t test_vcvtah_s64_f16 (float16_t a) {
  return vcvtah_s64_f16(a);
}

// CHECK-LABEL: test_vcvtah_u16_f16
// CHECK: [[FCVT:%.*]] = call i32 @llvm.aarch64.neon.fcvtau.i32.f16(half %a)
// CHECK: [[RET:%.*]] = trunc i32 [[FCVT]] to i16
// CHECK: ret i16 [[RET]]
uint16_t test_vcvtah_u16_f16 (float16_t a) {
  return vcvtah_u16_f16(a);
}

// CHECK-LABEL: test_vcvtah_u32_f16
// CHECK: [[VCVT:%.*]] = call i32 @llvm.aarch64.neon.fcvtau.i32.f16(half %a)
// CHECK: ret i32 [[VCVT]]
uint32_t test_vcvtah_u32_f16 (float16_t a) {
  return vcvtah_u32_f16(a);
}

// CHECK-LABEL: test_vcvtah_u64_f16
// CHECK: [[VCVT:%.*]] = call i64 @llvm.aarch64.neon.fcvtau.i64.f16(half %a)
// CHECK: ret i64 [[VCVT]]
uint64_t test_vcvtah_u64_f16 (float16_t a) {
  return vcvtah_u64_f16(a);
}

// CHECK-LABEL: test_vcvtmh_s16_f16
// CHECK: [[FCVT:%.*]] = call i32 @llvm.aarch64.neon.fcvtms.i32.f16(half %a)
// CHECK: [[RET:%.*]] = trunc i32 [[FCVT]] to i16
// CHECK: ret i16 [[RET]]
int16_t test_vcvtmh_s16_f16 (float16_t a) {
  return vcvtmh_s16_f16(a);
}

// CHECK-LABEL: test_vcvtmh_s32_f16
// CHECK: [[VCVT:%.*]] = call i32 @llvm.aarch64.neon.fcvtms.i32.f16(half %a)
// CHECK: ret i32 [[VCVT]]
int32_t test_vcvtmh_s32_f16 (float16_t a) {
  return vcvtmh_s32_f16(a);
}

// CHECK-LABEL: test_vcvtmh_s64_f16
// CHECK: [[VCVT:%.*]] = call i64 @llvm.aarch64.neon.fcvtms.i64.f16(half %a)
// CHECK: ret i64 [[VCVT]]
int64_t test_vcvtmh_s64_f16 (float16_t a) {
  return vcvtmh_s64_f16(a);
}

// CHECK-LABEL: test_vcvtmh_u16_f16
// CHECK: [[FCVT:%.*]] = call i32 @llvm.aarch64.neon.fcvtmu.i32.f16(half %a)
// CHECK: [[RET:%.*]] = trunc i32 [[FCVT]] to i16
// CHECK: ret i16 [[RET]]
uint16_t test_vcvtmh_u16_f16 (float16_t a) {
  return vcvtmh_u16_f16(a);
}

// CHECK-LABEL: test_vcvtmh_u32_f16
// CHECK: [[VCVT:%.*]] = call i32 @llvm.aarch64.neon.fcvtmu.i32.f16(half %a)
// CHECK: ret i32 [[VCVT]]
uint32_t test_vcvtmh_u32_f16 (float16_t a) {
  return vcvtmh_u32_f16(a);
}

// CHECK-LABEL: test_vcvtmh_u64_f16
// CHECK: [[VCVT:%.*]] = call i64 @llvm.aarch64.neon.fcvtmu.i64.f16(half %a)
// CHECK: ret i64 [[VCVT]]
uint64_t test_vcvtmh_u64_f16 (float16_t a) {
  return vcvtmh_u64_f16(a);
}

// CHECK-LABEL: test_vcvtnh_s16_f16
// CHECK: [[FCVT:%.*]] = call i32 @llvm.aarch64.neon.fcvtns.i32.f16(half %a)
// CHECK: [[RET:%.*]] = trunc i32 [[FCVT]] to i16
// CHECK: ret i16 [[RET]]
int16_t test_vcvtnh_s16_f16 (float16_t a) {
  return vcvtnh_s16_f16(a);
}

// CHECK-LABEL: test_vcvtnh_s32_f16
// CHECK: [[VCVT:%.*]] = call i32 @llvm.aarch64.neon.fcvtns.i32.f16(half %a)
// CHECK: ret i32 [[VCVT]]
int32_t test_vcvtnh_s32_f16 (float16_t a) {
  return vcvtnh_s32_f16(a);
}

// CHECK-LABEL: test_vcvtnh_s64_f16
// CHECK: [[VCVT:%.*]] = call i64 @llvm.aarch64.neon.fcvtns.i64.f16(half %a)
// CHECK: ret i64 [[VCVT]]
int64_t test_vcvtnh_s64_f16 (float16_t a) {
  return vcvtnh_s64_f16(a);
}

// CHECK-LABEL: test_vcvtnh_u16_f16
// CHECK: [[FCVT:%.*]] = call i32 @llvm.aarch64.neon.fcvtnu.i32.f16(half %a)
// CHECK: [[RET:%.*]] = trunc i32 [[FCVT]] to i16
// CHECK: ret i16 [[RET]]
uint16_t test_vcvtnh_u16_f16 (float16_t a) {
  return vcvtnh_u16_f16(a);
}

// CHECK-LABEL: test_vcvtnh_u32_f16
// CHECK: [[VCVT:%.*]] = call i32 @llvm.aarch64.neon.fcvtnu.i32.f16(half %a)
// CHECK: ret i32 [[VCVT]]
uint32_t test_vcvtnh_u32_f16 (float16_t a) {
  return vcvtnh_u32_f16(a);
}

// CHECK-LABEL: test_vcvtnh_u64_f16
// CHECK: [[VCVT:%.*]] = call i64 @llvm.aarch64.neon.fcvtnu.i64.f16(half %a)
// CHECK: ret i64 [[VCVT]]
uint64_t test_vcvtnh_u64_f16 (float16_t a) {
  return vcvtnh_u64_f16(a);
}

// CHECK-LABEL: test_vcvtph_s16_f16
// CHECK: [[FCVT:%.*]] = call i32 @llvm.aarch64.neon.fcvtps.i32.f16(half %a)
// CHECK: [[RET:%.*]] = trunc i32 [[FCVT]] to i16
// CHECK: ret i16 [[RET]]
int16_t test_vcvtph_s16_f16 (float16_t a) {
  return vcvtph_s16_f16(a);
}

// CHECK-LABEL: test_vcvtph_s32_f16
// CHECK: [[VCVT:%.*]] = call i32 @llvm.aarch64.neon.fcvtps.i32.f16(half %a)
// CHECK: ret i32 [[VCVT]]
int32_t test_vcvtph_s32_f16 (float16_t a) {
  return vcvtph_s32_f16(a);
}

// CHECK-LABEL: test_vcvtph_s64_f16
// CHECK: [[VCVT:%.*]] = call i64 @llvm.aarch64.neon.fcvtps.i64.f16(half %a)
// CHECK: ret i64 [[VCVT]]
int64_t test_vcvtph_s64_f16 (float16_t a) {
  return vcvtph_s64_f16(a);
}

// CHECK-LABEL: test_vcvtph_u16_f16
// CHECK: [[FCVT:%.*]] = call i32 @llvm.aarch64.neon.fcvtpu.i32.f16(half %a)
// CHECK: [[RET:%.*]] = trunc i32 [[FCVT]] to i16
// CHECK: ret i16 [[RET]]
uint16_t test_vcvtph_u16_f16 (float16_t a) {
  return vcvtph_u16_f16(a);
}

// CHECK-LABEL: test_vcvtph_u32_f16
// CHECK: [[VCVT:%.*]] = call i32 @llvm.aarch64.neon.fcvtpu.i32.f16(half %a)
// CHECK: ret i32 [[VCVT]]
uint32_t test_vcvtph_u32_f16 (float16_t a) {
  return vcvtph_u32_f16(a);
}

// CHECK-LABEL: test_vcvtph_u64_f16
// CHECK: [[VCVT:%.*]] = call i64 @llvm.aarch64.neon.fcvtpu.i64.f16(half %a)
// CHECK: ret i64 [[VCVT]]
uint64_t test_vcvtph_u64_f16 (float16_t a) {
  return vcvtph_u64_f16(a);
}

// CHECK-LABEL: test_vnegh_f16
// CHECK: [[NEG:%.*]] = fneg half %a
// CHECK: ret half [[NEG]]
float16_t test_vnegh_f16(float16_t a) {
  return vnegh_f16(a);
}

// CHECK-LABEL: test_vrecpeh_f16
// CHECK: [[VREC:%.*]] = call half @llvm.aarch64.neon.frecpe.f16(half %a)
// CHECK: ret half [[VREC]]
float16_t test_vrecpeh_f16(float16_t a) {
  return vrecpeh_f16(a);
}

// CHECK-LABEL: test_vrecpxh_f16
// CHECK: [[VREC:%.*]] = call half @llvm.aarch64.neon.frecpx.f16(half %a)
// CHECK: ret half [[VREC]]
float16_t test_vrecpxh_f16(float16_t a) {
  return vrecpxh_f16(a);
}

// CHECK-LABEL: test_vrndh_f16
// CHECK:  [[RND:%.*]] =  call half @llvm.trunc.f16(half %a)
// CHECK:  ret half [[RND]]
float16_t test_vrndh_f16(float16_t a) {
  return vrndh_f16(a);
}

// CHECK-LABEL: test_vrndah_f16
// CHECK:  [[RND:%.*]] =  call half @llvm.round.f16(half %a)
// CHECK:  ret half [[RND]]
float16_t test_vrndah_f16(float16_t a) {
  return vrndah_f16(a);
}

// CHECK-LABEL: test_vrndih_f16
// CHECK:  [[RND:%.*]] =  call half @llvm.nearbyint.f16(half %a)
// CHECK:  ret half [[RND]]
float16_t test_vrndih_f16(float16_t a) {
  return vrndih_f16(a);
}

// CHECK-LABEL: test_vrndmh_f16
// CHECK:  [[RND:%.*]] =  call half @llvm.floor.f16(half %a)
// CHECK:  ret half [[RND]]
float16_t test_vrndmh_f16(float16_t a) {
  return vrndmh_f16(a);
}

// CHECK-LABEL: test_vrndnh_f16
// CHECK:  [[RND:%.*]] =  call half @llvm.roundeven.f16(half %a)
// CHECK:  ret half [[RND]]
float16_t test_vrndnh_f16(float16_t a) {
  return vrndnh_f16(a);
}

// CHECK-LABEL: test_vrndph_f16
// CHECK:  [[RND:%.*]] =  call half @llvm.ceil.f16(half %a)
// CHECK:  ret half [[RND]]
float16_t test_vrndph_f16(float16_t a) {
  return vrndph_f16(a);
}

// CHECK-LABEL: test_vrndxh_f16
// CHECK:  [[RND:%.*]] =  call half @llvm.rint.f16(half %a)
// CHECK:  ret half [[RND]]
float16_t test_vrndxh_f16(float16_t a) {
  return vrndxh_f16(a);
}

// CHECK-LABEL: test_vrsqrteh_f16
// CHECK:  [[RND:%.*]] = call half @llvm.aarch64.neon.frsqrte.f16(half %a)
// CHECK:  ret half [[RND]]
float16_t test_vrsqrteh_f16(float16_t a) {
  return vrsqrteh_f16(a);
}

// CHECK-LABEL: test_vsqrth_f16
// CHECK:  [[SQR:%.*]] = call half @llvm.sqrt.f16(half %a)
// CHECK:  ret half [[SQR]]
float16_t test_vsqrth_f16(float16_t a) {
  return vsqrth_f16(a);
}

// CHECK-LABEL: test_vaddh_f16
// CHECK:  [[ADD:%.*]] = fadd half %a, %b
// CHECK:  ret half [[ADD]]
float16_t test_vaddh_f16(float16_t a, float16_t b) {
  return vaddh_f16(a, b);
}

// CHECK-LABEL: test_vabdh_f16
// CHECK:  [[ABD:%.*]] = call half @llvm.aarch64.sisd.fabd.f16(half %a, half %b)
// CHECK:  ret half [[ABD]]
float16_t test_vabdh_f16(float16_t a, float16_t b) {
  return vabdh_f16(a, b);
}

// CHECK-LABEL: test_vcageh_f16
// CHECK:  [[FACG:%.*]] = call i32 @llvm.aarch64.neon.facge.i32.f16(half %a, half %b)
// CHECK: [[RET:%.*]] = trunc i32 [[FACG]] to i16
// CHECK: ret i16 [[RET]]
uint16_t test_vcageh_f16(float16_t a, float16_t b) {
  return vcageh_f16(a, b);
}

// CHECK-LABEL: test_vcagth_f16
// CHECK:  [[FACG:%.*]] = call i32 @llvm.aarch64.neon.facgt.i32.f16(half %a, half %b)
// CHECK: [[RET:%.*]] = trunc i32 [[FACG]] to i16
// CHECK: ret i16 [[RET]]
uint16_t test_vcagth_f16(float16_t a, float16_t b) {
  return vcagth_f16(a, b);
}

// CHECK-LABEL: test_vcaleh_f16
// CHECK:  [[FACG:%.*]] = call i32 @llvm.aarch64.neon.facge.i32.f16(half %b, half %a)
// CHECK: [[RET:%.*]] = trunc i32 [[FACG]] to i16
// CHECK: ret i16 [[RET]]
uint16_t test_vcaleh_f16(float16_t a, float16_t b) {
  return vcaleh_f16(a, b);
}

// CHECK-LABEL: test_vcalth_f16
// CHECK:  [[FACG:%.*]] = call i32 @llvm.aarch64.neon.facgt.i32.f16(half %b, half %a)
// CHECK: [[RET:%.*]] = trunc i32 [[FACG]] to i16
// CHECK: ret i16 [[RET]]
uint16_t test_vcalth_f16(float16_t a, float16_t b) {
  return vcalth_f16(a, b);
}

// CHECK-LABEL: test_vceqh_f16
// CHECK:  [[TMP1:%.*]] = fcmp oeq half %a, %b
// CHECK:  [[TMP2:%.*]] = sext i1 [[TMP1]] to i16
// CHECK:  ret i16 [[TMP2]]
uint16_t test_vceqh_f16(float16_t a, float16_t b) {
  return vceqh_f16(a, b);
}

// CHECK-LABEL: test_vcgeh_f16
// CHECK:  [[TMP1:%.*]] = fcmp oge half %a, %b
// CHECK:  [[TMP2:%.*]] = sext i1 [[TMP1]] to i16
// CHECK:  ret i16 [[TMP2]]
uint16_t test_vcgeh_f16(float16_t a, float16_t b) {
  return vcgeh_f16(a, b);
}

// CHECK-LABEL: test_vcgth_f16
//CHECK:  [[TMP1:%.*]] = fcmp ogt half %a, %b
// CHECK:  [[TMP2:%.*]] = sext i1 [[TMP1]] to i16
// CHECK:  ret i16 [[TMP2]]
uint16_t test_vcgth_f16(float16_t a, float16_t b) {
  return vcgth_f16(a, b);
}

// CHECK-LABEL: test_vcleh_f16
// CHECK:  [[TMP1:%.*]] = fcmp ole half %a, %b
// CHECK:  [[TMP2:%.*]] = sext i1 [[TMP1]] to i16
// CHECK:  ret i16 [[TMP2]]
uint16_t test_vcleh_f16(float16_t a, float16_t b) {
  return vcleh_f16(a, b);
}

// CHECK-LABEL: test_vclth_f16
// CHECK:  [[TMP1:%.*]] = fcmp olt half %a, %b
// CHECK:  [[TMP2:%.*]] = sext i1 [[TMP1]] to i16
// CHECK:  ret i16 [[TMP2]]
uint16_t test_vclth_f16(float16_t a, float16_t b) {
  return vclth_f16(a, b);
}

// CHECK-LABEL: test_vcvth_n_f16_s16
// CHECK: [[SEXT:%.*]] = sext i16 %a to i32
// CHECK:  [[CVT:%.*]] = call half @llvm.aarch64.neon.vcvtfxs2fp.f16.i32(i32 [[SEXT]], i32 1)
// CHECK:  ret half [[CVT]]
float16_t test_vcvth_n_f16_s16(int16_t a) {
  return vcvth_n_f16_s16(a, 1);
}

// CHECK-LABEL: test_vcvth_n_f16_s32
// CHECK:  [[CVT:%.*]] = call half @llvm.aarch64.neon.vcvtfxs2fp.f16.i32(i32 %a, i32 1)
// CHECK:  ret half [[CVT]]
float16_t test_vcvth_n_f16_s32(int32_t a) {
  return vcvth_n_f16_s32(a, 1);
}

// CHECK-LABEL: test_vcvth_n_f16_s64
// CHECK:  [[CVT:%.*]] = call half @llvm.aarch64.neon.vcvtfxs2fp.f16.i64(i64 %a, i32 1)
// CHECK:  ret half [[CVT]]
float16_t test_vcvth_n_f16_s64(int64_t a) {
  return vcvth_n_f16_s64(a, 1);
}

// CHECK-LABEL: test_vcvth_n_s16_f16
// CHECK:  [[CVT:%.*]] = call i32 @llvm.aarch64.neon.vcvtfp2fxs.i32.f16(half %a, i32 1)
// CHECK: [[RET:%.*]] = trunc i32 [[CVT]] to i16
// CHECK: ret i16 [[RET]]
int16_t test_vcvth_n_s16_f16(float16_t a) {
  return vcvth_n_s16_f16(a, 1);
}

// CHECK-LABEL: test_vcvth_n_s32_f16
// CHECK:  [[CVT:%.*]] = call i32 @llvm.aarch64.neon.vcvtfp2fxs.i32.f16(half %a, i32 1)
// CHECK:  ret i32 [[CVT]]
int32_t test_vcvth_n_s32_f16(float16_t a) {
  return vcvth_n_s32_f16(a, 1);
}

// CHECK-LABEL: test_vcvth_n_s64_f16
// CHECK:  [[CVT:%.*]] = call i64 @llvm.aarch64.neon.vcvtfp2fxs.i64.f16(half %a, i32 1)
// CHECK:  ret i64 [[CVT]]
int64_t test_vcvth_n_s64_f16(float16_t a) {
  return vcvth_n_s64_f16(a, 1);
}

// CHECK-LABEL: test_vcvth_n_f16_u16
// CHECK: [[SEXT:%.*]] = zext i16 %a to i32
// CHECK:  [[CVT:%.*]] = call half @llvm.aarch64.neon.vcvtfxu2fp.f16.i32(i32 [[SEXT]], i32 1)
// CHECK:  ret half [[CVT]]
float16_t test_vcvth_n_f16_u16(int16_t a) {
  return vcvth_n_f16_u16(a, 1);
}

// CHECK-LABEL: test_vcvth_n_f16_u32
// CHECK:  [[CVT:%.*]] = call half @llvm.aarch64.neon.vcvtfxu2fp.f16.i32(i32 %a, i32 1)
// CHECK:  ret half [[CVT]]
float16_t test_vcvth_n_f16_u32(int32_t a) {
  return vcvth_n_f16_u32(a, 1);
}

// CHECK-LABEL: test_vcvth_n_f16_u64
// CHECK:  [[CVT:%.*]] = call half @llvm.aarch64.neon.vcvtfxu2fp.f16.i64(i64 %a, i32 1)
// CHECK:  ret half [[CVT]]
float16_t test_vcvth_n_f16_u64(int64_t a) {
  return vcvth_n_f16_u64(a, 1);
}

// CHECK-LABEL: test_vcvth_n_u16_f16
// CHECK:  [[CVT:%.*]] = call i32 @llvm.aarch64.neon.vcvtfp2fxu.i32.f16(half %a, i32 1)
// CHECK: [[RET:%.*]] = trunc i32 [[CVT]] to i16
// CHECK: ret i16 [[RET]]
int16_t test_vcvth_n_u16_f16(float16_t a) {
  return vcvth_n_u16_f16(a, 1);
}

// CHECK-LABEL: test_vcvth_n_u32_f16
// CHECK:  [[CVT:%.*]] = call i32 @llvm.aarch64.neon.vcvtfp2fxu.i32.f16(half %a, i32 1)
// CHECK:  ret i32 [[CVT]]
int32_t test_vcvth_n_u32_f16(float16_t a) {
  return vcvth_n_u32_f16(a, 1);
}

// CHECK-LABEL: test_vcvth_n_u64_f16
// CHECK:  [[CVT:%.*]] = call i64 @llvm.aarch64.neon.vcvtfp2fxu.i64.f16(half %a, i32 1)
// CHECK:  ret i64 [[CVT]]
int64_t test_vcvth_n_u64_f16(float16_t a) {
  return vcvth_n_u64_f16(a, 1);
}

// CHECK-LABEL: test_vdivh_f16
// CHECK:  [[DIV:%.*]] = fdiv half %a, %b
// CHECK:  ret half [[DIV]]
float16_t test_vdivh_f16(float16_t a, float16_t b) {
  return vdivh_f16(a, b);
}

// CHECK-LABEL: test_vmaxh_f16
// CHECK:  [[MAX:%.*]] = call half @llvm.aarch64.neon.fmax.f16(half %a, half %b)
// CHECK:  ret half [[MAX]]
float16_t test_vmaxh_f16(float16_t a, float16_t b) {
  return vmaxh_f16(a, b);
}

// CHECK-LABEL: test_vmaxnmh_f16
// CHECK:  [[MAX:%.*]] = call half @llvm.aarch64.neon.fmaxnm.f16(half %a, half %b)
// CHECK:  ret half [[MAX]]
float16_t test_vmaxnmh_f16(float16_t a, float16_t b) {
  return vmaxnmh_f16(a, b);
}

// CHECK-LABEL: test_vminh_f16
// CHECK:  [[MIN:%.*]] = call half @llvm.aarch64.neon.fmin.f16(half %a, half %b)
// CHECK:  ret half [[MIN]]
float16_t test_vminh_f16(float16_t a, float16_t b) {
  return vminh_f16(a, b);
}

// CHECK-LABEL: test_vminnmh_f16
// CHECK:  [[MIN:%.*]] = call half @llvm.aarch64.neon.fminnm.f16(half %a, half %b)
// CHECK:  ret half [[MIN]]
float16_t test_vminnmh_f16(float16_t a, float16_t b) {
  return vminnmh_f16(a, b);
}

// CHECK-LABEL: test_vmulh_f16
// CHECK:  [[MUL:%.*]] = fmul half %a, %b
// CHECK:  ret half [[MUL]]
float16_t test_vmulh_f16(float16_t a, float16_t b) {
  return vmulh_f16(a, b);
}

// CHECK-LABEL: test_vmulxh_f16
// CHECK:  [[MUL:%.*]] = call half @llvm.aarch64.neon.fmulx.f16(half %a, half %b)
// CHECK:  ret half [[MUL]]
float16_t test_vmulxh_f16(float16_t a, float16_t b) {
  return vmulxh_f16(a, b);
}

// CHECK-LABEL: test_vrecpsh_f16
// CHECK: [[RECPS:%.*]] = call half @llvm.aarch64.neon.frecps.f16(half %a, half %b)
// CHECK: ret half [[RECPS]]
float16_t test_vrecpsh_f16(float16_t a, float16_t b) {
  return vrecpsh_f16(a, b);
}

// CHECK-LABEL: test_vrsqrtsh_f16
// CHECK:  [[RSQRTS:%.*]] = call half @llvm.aarch64.neon.frsqrts.f16(half %a, half %b)
// CHECK:  ret half [[RSQRTS]]
float16_t test_vrsqrtsh_f16(float16_t a, float16_t b) {
  return vrsqrtsh_f16(a, b);
}

// CHECK-LABEL: test_vsubh_f16
// CHECK:  [[SUB:%.*]] = fsub half %a, %b
// CHECK:  ret half [[SUB]]
float16_t test_vsubh_f16(float16_t a, float16_t b) {
  return vsubh_f16(a, b);
}

// CHECK-LABEL: test_vfmah_f16
// CHECK:  [[FMA:%.*]] = call half @llvm.fma.f16(half %b, half %c, half %a)
// CHECK:  ret half [[FMA]]
float16_t test_vfmah_f16(float16_t a, float16_t b, float16_t c) {
  return vfmah_f16(a, b, c);
}

// CHECK-LABEL: test_vfmsh_f16
// CHECK:  [[SUB:%.*]] = fsub half 0xH8000, %b
// CHECK:  [[ADD:%.*]] = call half @llvm.fma.f16(half [[SUB]], half %c, half %a)
// CHECK:  ret half [[ADD]]
float16_t test_vfmsh_f16(float16_t a, float16_t b, float16_t c) {
  return vfmsh_f16(a, b, c);
}

