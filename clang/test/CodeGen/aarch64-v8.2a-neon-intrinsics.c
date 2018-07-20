// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +fullfp16 -target-feature +v8.2a\
// RUN: -fallow-half-arguments-and-returns -S -disable-O0-optnone -emit-llvm -o - %s \
// RUN: | opt -S -mem2reg \
// RUN: | FileCheck %s

// REQUIRES: aarch64-registered-target

#include <arm_neon.h>

// CHECK-LABEL: test_vabs_f16
// CHECK:  [[ABS:%.*]] =  call <4 x half> @llvm.fabs.v4f16(<4 x half> %a)
// CHECK:  ret <4 x half> [[ABS]]
float16x4_t test_vabs_f16(float16x4_t a) {
  return vabs_f16(a);
}

// CHECK-LABEL: test_vabsq_f16
// CHECK:  [[ABS:%.*]] = call <8 x half> @llvm.fabs.v8f16(<8 x half> %a)
// CHECK:  ret <8 x half> [[ABS]]
float16x8_t test_vabsq_f16(float16x8_t a) {
  return vabsq_f16(a);
}

// CHECK-LABEL: test_vceqz_f16
// CHECK:  [[TMP1:%.*]] = fcmp oeq <4 x half> %a, zeroinitializer
// CHECK:  [[TMP2:%.*]] = sext <4 x i1> [[TMP1]] to <4 x i16>
// CHECK:  ret <4 x i16> [[TMP2]]
uint16x4_t test_vceqz_f16(float16x4_t a) {
  return vceqz_f16(a);
}

// CHECK-LABEL: test_vceqzq_f16
// CHECK:  [[TMP1:%.*]] = fcmp oeq <8 x half> %a, zeroinitializer
// CHECK:  [[TMP2:%.*]] = sext <8 x i1> [[TMP1:%.*]] to <8 x i16>
// CHECK:  ret <8 x i16> [[TMP2]]
uint16x8_t test_vceqzq_f16(float16x8_t a) {
  return vceqzq_f16(a);
}

// CHECK-LABEL: test_vcgez_f16
// CHECK:  [[TMP1:%.*]] = fcmp oge <4 x half> %a, zeroinitializer
// CHECK:  [[TMP2:%.*]] = sext <4 x i1> [[TMP1]] to <4 x i16>
// CHECK:  ret <4 x i16> [[TMP2]]
uint16x4_t test_vcgez_f16(float16x4_t a) {
  return vcgez_f16(a);
}

// CHECK-LABEL: test_vcgezq_f16
// CHECK:  [[TMP1:%.*]] = fcmp oge <8 x half> %a, zeroinitializer
// CHECK:  [[TMP2:%.*]] = sext <8 x i1> [[TMP1:%.*]] to <8 x i16>
// CHECK:  ret <8 x i16> [[TMP2]]
uint16x8_t test_vcgezq_f16(float16x8_t a) {
  return vcgezq_f16(a);
}

// CHECK-LABEL: test_vcgtz_f16
// CHECK:  [[TMP1:%.*]] = fcmp ogt <4 x half> %a, zeroinitializer
// CHECK:  [[TMP2:%.*]] = sext <4 x i1> [[TMP1]] to <4 x i16>
// CHECK:  ret <4 x i16> [[TMP2]]
uint16x4_t test_vcgtz_f16(float16x4_t a) {
  return vcgtz_f16(a);
}

// CHECK-LABEL: test_vcgtzq_f16
// CHECK:  [[TMP1:%.*]] = fcmp ogt <8 x half> %a, zeroinitializer
// CHECK:  [[TMP2:%.*]] = sext <8 x i1> [[TMP1:%.*]] to <8 x i16>
// CHECK:  ret <8 x i16> [[TMP2]]
uint16x8_t test_vcgtzq_f16(float16x8_t a) {
  return vcgtzq_f16(a);
}

// CHECK-LABEL: test_vclez_f16
// CHECK:  [[TMP1:%.*]] = fcmp ole <4 x half> %a, zeroinitializer
// CHECK:  [[TMP2:%.*]] = sext <4 x i1> [[TMP1]] to <4 x i16>
// CHECK:  ret <4 x i16> [[TMP2]]
uint16x4_t test_vclez_f16(float16x4_t a) {
  return vclez_f16(a);
}

// CHECK-LABEL: test_vclezq_f16
// CHECK:  [[TMP1:%.*]] = fcmp ole <8 x half> %a, zeroinitializer
// CHECK:  [[TMP2:%.*]] = sext <8 x i1> [[TMP1:%.*]] to <8 x i16>
// CHECK:  ret <8 x i16> [[TMP2]]
uint16x8_t test_vclezq_f16(float16x8_t a) {
  return vclezq_f16(a);
}

// CHECK-LABEL: test_vcltz_f16
// CHECK:  [[TMP1:%.*]] = fcmp olt <4 x half> %a, zeroinitializer
// CHECK:  [[TMP2:%.*]] = sext <4 x i1> [[TMP1]] to <4 x i16>
// CHECK:  ret <4 x i16> [[TMP2]]
uint16x4_t test_vcltz_f16(float16x4_t a) {
  return vcltz_f16(a);
}

// CHECK-LABEL: test_vcltzq_f16
// CHECK:  [[TMP1:%.*]] = fcmp olt <8 x half> %a, zeroinitializer
// CHECK:  [[TMP2:%.*]] = sext <8 x i1> [[TMP1:%.*]] to <8 x i16>
// CHECK:  ret <8 x i16> [[TMP2]]
uint16x8_t test_vcltzq_f16(float16x8_t a) {
  return vcltzq_f16(a);
}

// CHECK-LABEL: test_vcvt_f16_s16
// CHECK:  [[VCVT:%.*]] = sitofp <4 x i16> %a to <4 x half>
// CHECK:  ret <4 x half> [[VCVT]]
float16x4_t test_vcvt_f16_s16 (int16x4_t a) {
  return vcvt_f16_s16(a);
}

// CHECK-LABEL: test_vcvtq_f16_s16
// CHECK:  [[VCVT:%.*]] = sitofp <8 x i16> %a to <8 x half>
// CHECK:  ret <8 x half> [[VCVT]]
float16x8_t test_vcvtq_f16_s16 (int16x8_t a) {
  return vcvtq_f16_s16(a);
}

// CHECK-LABEL: test_vcvt_f16_u16
// CHECK:  [[VCVT:%.*]] = uitofp <4 x i16> %a to <4 x half>
// CHECK:  ret <4 x half> [[VCVT]]
float16x4_t test_vcvt_f16_u16 (uint16x4_t a) {
  return vcvt_f16_u16(a);
}

// CHECK-LABEL: test_vcvtq_f16_u16
// CHECK:  [[VCVT:%.*]] = uitofp <8 x i16> %a to <8 x half>
// CHECK:  ret <8 x half> [[VCVT]]
float16x8_t test_vcvtq_f16_u16 (uint16x8_t a) {
  return vcvtq_f16_u16(a);
}

// CHECK-LABEL: test_vcvt_s16_f16
// CHECK:  [[VCVT:%.*]] = fptosi <4 x half> %a to <4 x i16>
// CHECK:  ret <4 x i16> [[VCVT]]
int16x4_t test_vcvt_s16_f16 (float16x4_t a) {
  return vcvt_s16_f16(a);
}

// CHECK-LABEL: test_vcvtq_s16_f16
// CHECK:  [[VCVT:%.*]] = fptosi <8 x half> %a to <8 x i16>
// CHECK:  ret <8 x i16> [[VCVT]]
int16x8_t test_vcvtq_s16_f16 (float16x8_t a) {
  return vcvtq_s16_f16(a);
}

// CHECK-LABEL: test_vcvt_u16_f16
// CHECK:  [[VCVT:%.*]] = fptoui <4 x half> %a to <4 x i16>
// CHECK:  ret <4 x i16> [[VCVT]]
int16x4_t test_vcvt_u16_f16 (float16x4_t a) {
  return vcvt_u16_f16(a);
}

// CHECK-LABEL: test_vcvtq_u16_f16
// CHECK:  [[VCVT:%.*]] = fptoui <8 x half> %a to <8 x i16>
// CHECK:  ret <8 x i16> [[VCVT]]
int16x8_t test_vcvtq_u16_f16 (float16x8_t a) {
  return vcvtq_u16_f16(a);
}

// CHECK-LABEL: test_vcvta_s16_f16
// CHECK:  [[VCVT:%.*]] = call <4 x i16> @llvm.aarch64.neon.fcvtas.v4i16.v4f16(<4 x half> %a)
// CHECK:  ret <4 x i16> [[VCVT]]
int16x4_t test_vcvta_s16_f16 (float16x4_t a) {
  return vcvta_s16_f16(a);
}

// CHECK-LABEL: test_vcvta_u16_f16
// CHECK:  [[VCVT:%.*]] = call <4 x i16> @llvm.aarch64.neon.fcvtau.v4i16.v4f16(<4 x half> %a)
// CHECK:  ret <4 x i16> [[VCVT]]
int16x4_t test_vcvta_u16_f16 (float16x4_t a) {
  return vcvta_u16_f16(a);
}

// CHECK-LABEL: test_vcvtaq_s16_f16
// CHECK:  [[VCVT:%.*]] = call <8 x i16> @llvm.aarch64.neon.fcvtas.v8i16.v8f16(<8 x half> %a)
// CHECK:  ret <8 x i16> [[VCVT]]
int16x8_t test_vcvtaq_s16_f16 (float16x8_t a) {
  return vcvtaq_s16_f16(a);
}

// CHECK-LABEL: test_vcvtm_s16_f16
// CHECK:  [[VCVT:%.*]] = call <4 x i16> @llvm.aarch64.neon.fcvtms.v4i16.v4f16(<4 x half> %a)
// CHECK:  ret <4 x i16> [[VCVT]]
int16x4_t test_vcvtm_s16_f16 (float16x4_t a) {
  return vcvtm_s16_f16(a);
}

// CHECK-LABEL: test_vcvtmq_s16_f16
// CHECK:  [[VCVT:%.*]] = call <8 x i16> @llvm.aarch64.neon.fcvtms.v8i16.v8f16(<8 x half> %a)
// CHECK:  ret <8 x i16> [[VCVT]]
int16x8_t test_vcvtmq_s16_f16 (float16x8_t a) {
  return vcvtmq_s16_f16(a);
}

// CHECK-LABEL: test_vcvtm_u16_f16
// CHECK:  [[VCVT:%.*]] = call <4 x i16> @llvm.aarch64.neon.fcvtmu.v4i16.v4f16(<4 x half> %a)
// CHECK:  ret <4 x i16> [[VCVT]]
uint16x4_t test_vcvtm_u16_f16 (float16x4_t a) {
  return vcvtm_u16_f16(a);
}

// CHECK-LABEL: test_vcvtmq_u16_f16
// CHECK:  [[VCVT:%.*]] = call <8 x i16> @llvm.aarch64.neon.fcvtmu.v8i16.v8f16(<8 x half> %a)
// CHECK:  ret <8 x i16> [[VCVT]]
uint16x8_t test_vcvtmq_u16_f16 (float16x8_t a) {
  return vcvtmq_u16_f16(a);
}

// CHECK-LABEL: test_vcvtn_s16_f16
// CHECK:  [[VCVT:%.*]] = call <4 x i16> @llvm.aarch64.neon.fcvtns.v4i16.v4f16(<4 x half> %a)
// CHECK:  ret <4 x i16> [[VCVT]]
int16x4_t test_vcvtn_s16_f16 (float16x4_t a) {
  return vcvtn_s16_f16(a);
}

// CHECK-LABEL: test_vcvtnq_s16_f16
// CHECK:  [[VCVT:%.*]] = call <8 x i16> @llvm.aarch64.neon.fcvtns.v8i16.v8f16(<8 x half> %a)
// CHECK:  ret <8 x i16> [[VCVT]]
int16x8_t test_vcvtnq_s16_f16 (float16x8_t a) {
  return vcvtnq_s16_f16(a);
}

// CHECK-LABEL: test_vcvtn_u16_f16
// CHECK:  [[VCVT:%.*]] = call <4 x i16> @llvm.aarch64.neon.fcvtnu.v4i16.v4f16(<4 x half> %a)
// CHECK:  ret <4 x i16> [[VCVT]]
uint16x4_t test_vcvtn_u16_f16 (float16x4_t a) {
  return vcvtn_u16_f16(a);
}

// CHECK-LABEL: test_vcvtnq_u16_f16
// CHECK:  [[VCVT:%.*]] = call <8 x i16> @llvm.aarch64.neon.fcvtnu.v8i16.v8f16(<8 x half> %a)
// CHECK:  ret <8 x i16> [[VCVT]]
uint16x8_t test_vcvtnq_u16_f16 (float16x8_t a) {
  return vcvtnq_u16_f16(a);
}

// CHECK-LABEL: test_vcvtp_s16_f16
// CHECK:  [[VCVT:%.*]] = call <4 x i16> @llvm.aarch64.neon.fcvtps.v4i16.v4f16(<4 x half> %a)
// CHECK:  ret <4 x i16> [[VCVT]]
int16x4_t test_vcvtp_s16_f16 (float16x4_t a) {
  return vcvtp_s16_f16(a);
}

// CHECK-LABEL: test_vcvtpq_s16_f16
// CHECK:  [[VCVT:%.*]] = call <8 x i16> @llvm.aarch64.neon.fcvtps.v8i16.v8f16(<8 x half> %a)
// CHECK:  ret <8 x i16> [[VCVT]]
int16x8_t test_vcvtpq_s16_f16 (float16x8_t a) {
  return vcvtpq_s16_f16(a);
}

// CHECK-LABEL: test_vcvtp_u16_f16
// CHECK:  [[VCVT:%.*]] = call <4 x i16> @llvm.aarch64.neon.fcvtpu.v4i16.v4f16(<4 x half> %a)
// CHECK:  ret <4 x i16> [[VCVT]]
uint16x4_t test_vcvtp_u16_f16 (float16x4_t a) {
  return vcvtp_u16_f16(a);
}

// CHECK-LABEL: test_vcvtpq_u16_f16
// CHECK:  [[VCVT:%.*]] = call <8 x i16> @llvm.aarch64.neon.fcvtpu.v8i16.v8f16(<8 x half> %a)
// CHECK:  ret <8 x i16> [[VCVT]]
uint16x8_t test_vcvtpq_u16_f16 (float16x8_t a) {
  return vcvtpq_u16_f16(a);
}

// FIXME: Fix the zero constant when fp16 non-storage-only type becomes available.
// CHECK-LABEL: test_vneg_f16
// CHECK:  [[NEG:%.*]] = fsub <4 x half> <half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000>, %a
// CHECK:  ret <4 x half> [[NEG]]
float16x4_t test_vneg_f16(float16x4_t a) {
  return vneg_f16(a);
}

// CHECK-LABEL: test_vnegq_f16
// CHECK:  [[NEG:%.*]] = fsub <8 x half> <half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000>, %a
// CHECK:  ret <8 x half> [[NEG]]
float16x8_t test_vnegq_f16(float16x8_t a) {
  return vnegq_f16(a);
}

// CHECK-LABEL: test_vrecpe_f16
// CHECK:  [[RCP:%.*]] = call <4 x half> @llvm.aarch64.neon.frecpe.v4f16(<4 x half> %a)
// CHECK:  ret <4 x half> [[RCP]]
float16x4_t test_vrecpe_f16(float16x4_t a) {
  return vrecpe_f16(a);
}

// CHECK-LABEL: test_vrecpeq_f16
// CHECK:  [[RCP:%.*]] = call <8 x half> @llvm.aarch64.neon.frecpe.v8f16(<8 x half> %a)
// CHECK:  ret <8 x half> [[RCP]]
float16x8_t test_vrecpeq_f16(float16x8_t a) {
  return vrecpeq_f16(a);
}

// CHECK-LABEL: test_vrnd_f16
// CHECK:  [[RND:%.*]] =  call <4 x half> @llvm.trunc.v4f16(<4 x half> %a)
// CHECK:  ret <4 x half> [[RND]]
float16x4_t test_vrnd_f16(float16x4_t a) {
  return vrnd_f16(a);
}

// CHECK-LABEL: test_vrndq_f16
// CHECK:  [[RND:%.*]] =  call <8 x half> @llvm.trunc.v8f16(<8 x half> %a)
// CHECK:  ret <8 x half> [[RND]]
float16x8_t test_vrndq_f16(float16x8_t a) {
  return vrndq_f16(a);
}

// CHECK-LABEL: test_vrnda_f16
// CHECK:  [[RND:%.*]] =  call <4 x half> @llvm.round.v4f16(<4 x half> %a)
// CHECK:  ret <4 x half> [[RND]]
float16x4_t test_vrnda_f16(float16x4_t a) {
  return vrnda_f16(a);
}

// CHECK-LABEL: test_vrndaq_f16
// CHECK:  [[RND:%.*]] =  call <8 x half> @llvm.round.v8f16(<8 x half> %a)
// CHECK:  ret <8 x half> [[RND]]
float16x8_t test_vrndaq_f16(float16x8_t a) {
  return vrndaq_f16(a);
}

// CHECK-LABEL: test_vrndi_f16
// CHECK:  [[RND:%.*]] =  call <4 x half> @llvm.nearbyint.v4f16(<4 x half> %a)
// CHECK:  ret <4 x half> [[RND]]
float16x4_t test_vrndi_f16(float16x4_t a) {
  return vrndi_f16(a);
}

// CHECK-LABEL: test_vrndiq_f16
// CHECK:  [[RND:%.*]] =  call <8 x half> @llvm.nearbyint.v8f16(<8 x half> %a)
// CHECK:  ret <8 x half> [[RND]]
float16x8_t test_vrndiq_f16(float16x8_t a) {
  return vrndiq_f16(a);
}

// CHECK-LABEL: test_vrndm_f16
// CHECK:  [[RND:%.*]] =  call <4 x half> @llvm.floor.v4f16(<4 x half> %a)
// CHECK:  ret <4 x half> [[RND]]
float16x4_t test_vrndm_f16(float16x4_t a) {
  return vrndm_f16(a);
}

// CHECK-LABEL: test_vrndmq_f16
// CHECK:  [[RND:%.*]] =  call <8 x half> @llvm.floor.v8f16(<8 x half> %a)
// CHECK:  ret <8 x half> [[RND]]
float16x8_t test_vrndmq_f16(float16x8_t a) {
  return vrndmq_f16(a);
}

// CHECK-LABEL: test_vrndn_f16
// CHECK:  [[RND:%.*]] =  call <4 x half> @llvm.aarch64.neon.frintn.v4f16(<4 x half> %a)
// CHECK:  ret <4 x half> [[RND]]
float16x4_t test_vrndn_f16(float16x4_t a) {
  return vrndn_f16(a);
}

// CHECK-LABEL: test_vrndnq_f16
// CHECK:  [[RND:%.*]] =  call <8 x half> @llvm.aarch64.neon.frintn.v8f16(<8 x half> %a)
// CHECK:  ret <8 x half> [[RND]]
float16x8_t test_vrndnq_f16(float16x8_t a) {
  return vrndnq_f16(a);
}

// CHECK-LABEL: test_vrndp_f16
// CHECK:  [[RND:%.*]] =  call <4 x half> @llvm.ceil.v4f16(<4 x half> %a)
// CHECK:  ret <4 x half> [[RND]]
float16x4_t test_vrndp_f16(float16x4_t a) {
  return vrndp_f16(a);
}

// CHECK-LABEL: test_vrndpq_f16
// CHECK:  [[RND:%.*]] =  call <8 x half> @llvm.ceil.v8f16(<8 x half> %a)
// CHECK:  ret <8 x half> [[RND]]
float16x8_t test_vrndpq_f16(float16x8_t a) {
  return vrndpq_f16(a);
}

// CHECK-LABEL: test_vrndx_f16
// CHECK:  [[RND:%.*]] =  call <4 x half> @llvm.rint.v4f16(<4 x half> %a)
// CHECK:  ret <4 x half> [[RND]]
float16x4_t test_vrndx_f16(float16x4_t a) {
  return vrndx_f16(a);
}

// CHECK-LABEL: test_vrndxq_f16
// CHECK:  [[RND:%.*]] =  call <8 x half> @llvm.rint.v8f16(<8 x half> %a)
// CHECK:  ret <8 x half> [[RND]]
float16x8_t test_vrndxq_f16(float16x8_t a) {
  return vrndxq_f16(a);
}

// CHECK-LABEL: test_vrsqrte_f16
// CHECK:  [[RND:%.*]] = call <4 x half> @llvm.aarch64.neon.frsqrte.v4f16(<4 x half> %a)
// CHECK:  ret <4 x half> [[RND]]
float16x4_t test_vrsqrte_f16(float16x4_t a) {
  return vrsqrte_f16(a);
}

// CHECK-LABEL: test_vrsqrteq_f16
// CHECK:  [[RND:%.*]] = call <8 x half> @llvm.aarch64.neon.frsqrte.v8f16(<8 x half> %a)
// CHECK:  ret <8 x half> [[RND]]
float16x8_t test_vrsqrteq_f16(float16x8_t a) {
  return vrsqrteq_f16(a);
}

// CHECK-LABEL: test_vsqrt_f16
// CHECK:  [[SQR:%.*]] = call <4 x half> @llvm.sqrt.v4f16(<4 x half> %a)
// CHECK:  ret <4 x half> [[SQR]]
float16x4_t test_vsqrt_f16(float16x4_t a) {
  return vsqrt_f16(a);
}

// CHECK-LABEL: test_vsqrtq_f16
// CHECK:  [[SQR:%.*]] = call <8 x half> @llvm.sqrt.v8f16(<8 x half> %a)
// CHECK:  ret <8 x half> [[SQR]]
float16x8_t test_vsqrtq_f16(float16x8_t a) {
  return vsqrtq_f16(a);
}

// CHECK-LABEL: test_vadd_f16
// CHECK:  [[ADD:%.*]] = fadd <4 x half> %a, %b
// CHECK:  ret <4 x half> [[ADD]]
float16x4_t test_vadd_f16(float16x4_t a, float16x4_t b) {
  return vadd_f16(a, b);
}

// CHECK-LABEL: test_vaddq_f16
// CHECK:  [[ADD:%.*]] = fadd <8 x half> %a, %b
// CHECK:  ret <8 x half> [[ADD]]
float16x8_t test_vaddq_f16(float16x8_t a, float16x8_t b) {
  return vaddq_f16(a, b);
}

// CHECK-LABEL: test_vabd_f16
// CHECK:  [[ABD:%.*]] = call <4 x half> @llvm.aarch64.neon.fabd.v4f16(<4 x half> %a, <4 x half> %b)
// CHECK:  ret <4 x half> [[ABD]]
float16x4_t test_vabd_f16(float16x4_t a, float16x4_t b) {
  return vabd_f16(a, b);
}

// CHECK-LABEL: test_vabdq_f16
// CHECK:  [[ABD:%.*]] = call <8 x half> @llvm.aarch64.neon.fabd.v8f16(<8 x half> %a, <8 x half> %b)
// CHECK:  ret <8 x half> [[ABD]]
float16x8_t test_vabdq_f16(float16x8_t a, float16x8_t b) {
  return vabdq_f16(a, b);
}

// CHECK-LABEL: test_vcage_f16
// CHECK:  [[ABS:%.*]] = call <4 x i16> @llvm.aarch64.neon.facge.v4i16.v4f16(<4 x half> %a, <4 x half> %b)
// CHECK:  ret <4 x i16> [[ABS]]
uint16x4_t test_vcage_f16(float16x4_t a, float16x4_t b) {
  return vcage_f16(a, b);
}

// CHECK-LABEL: test_vcageq_f16
// CHECK:  [[ABS:%.*]] = call <8 x i16> @llvm.aarch64.neon.facge.v8i16.v8f16(<8 x half> %a, <8 x half> %b)
// CHECK:  ret <8 x i16> [[ABS]]
uint16x8_t test_vcageq_f16(float16x8_t a, float16x8_t b) {
  return vcageq_f16(a, b);
}

// CHECK-LABEL: test_vcagt_f16
// CHECK:  [[ABS:%.*]] = call <4 x i16> @llvm.aarch64.neon.facgt.v4i16.v4f16(<4 x half> %a, <4 x half> %b)
// CHECK:  ret <4 x i16> [[ABS]]
uint16x4_t test_vcagt_f16(float16x4_t a, float16x4_t b) {
  return vcagt_f16(a, b);
}

// CHECK-LABEL: test_vcagtq_f16
// CHECK:  [[ABS:%.*]] = call <8 x i16> @llvm.aarch64.neon.facgt.v8i16.v8f16(<8 x half> %a, <8 x half> %b)
// CHECK:  ret <8 x i16> [[ABS]]
uint16x8_t test_vcagtq_f16(float16x8_t a, float16x8_t b) {
  return vcagtq_f16(a, b);
}

// CHECK-LABEL: test_vcale_f16
// CHECK:  [[ABS:%.*]] = call <4 x i16> @llvm.aarch64.neon.facge.v4i16.v4f16(<4 x half> %b, <4 x half> %a)
// CHECK:  ret <4 x i16> [[ABS]]
uint16x4_t test_vcale_f16(float16x4_t a, float16x4_t b) {
  return vcale_f16(a, b);
}

// CHECK-LABEL: test_vcaleq_f16
// CHECK:  [[ABS:%.*]] = call <8 x i16> @llvm.aarch64.neon.facge.v8i16.v8f16(<8 x half> %b, <8 x half> %a)
// CHECK:  ret <8 x i16> [[ABS]]
uint16x8_t test_vcaleq_f16(float16x8_t a, float16x8_t b) {
  return vcaleq_f16(a, b);
}

// CHECK-LABEL: test_vcalt_f16
// CHECK:  [[ABS:%.*]] = call <4 x i16> @llvm.aarch64.neon.facgt.v4i16.v4f16(<4 x half> %b, <4 x half> %a)
// CHECK:  ret <4 x i16> [[ABS]]
uint16x4_t test_vcalt_f16(float16x4_t a, float16x4_t b) {
  return vcalt_f16(a, b);
}

// CHECK-LABEL: test_vcaltq_f16
// CHECK:  [[ABS:%.*]] = call <8 x i16> @llvm.aarch64.neon.facgt.v8i16.v8f16(<8 x half> %b, <8 x half> %a)
// CHECK:  ret <8 x i16> [[ABS]]
uint16x8_t test_vcaltq_f16(float16x8_t a, float16x8_t b) {
  return vcaltq_f16(a, b);
}

// CHECK-LABEL: test_vceq_f16
// CHECK:  [[TMP1:%.*]] = fcmp oeq <4 x half> %a, %b
// CHECK:  [[TMP2:%.*]] = sext <4 x i1> [[TMP1]] to <4 x i16>
// CHECK:  ret <4 x i16> [[TMP2]]
uint16x4_t test_vceq_f16(float16x4_t a, float16x4_t b) {
  return vceq_f16(a, b);
}

// CHECK-LABEL: test_vceqq_f16
// CHECK:  [[TMP1:%.*]] = fcmp oeq <8 x half> %a, %b
// CHECK:  [[TMP2:%.*]] = sext <8 x i1> [[TMP1:%.*]] to <8 x i16>
// CHECK:  ret <8 x i16> [[TMP2]]
uint16x8_t test_vceqq_f16(float16x8_t a, float16x8_t b) {
  return vceqq_f16(a, b);
}

// CHECK-LABEL: test_vcge_f16
// CHECK:  [[TMP1:%.*]] = fcmp oge <4 x half> %a, %b
// CHECK:  [[TMP2:%.*]] = sext <4 x i1> [[TMP1]] to <4 x i16>
// CHECK:  ret <4 x i16> [[TMP2]]
uint16x4_t test_vcge_f16(float16x4_t a, float16x4_t b) {
  return vcge_f16(a, b);
}

// CHECK-LABEL: test_vcgeq_f16
// CHECK:  [[TMP1:%.*]] = fcmp oge <8 x half> %a, %b
// CHECK:  [[TMP2:%.*]] = sext <8 x i1> [[TMP1:%.*]] to <8 x i16>
// CHECK:  ret <8 x i16> [[TMP2]]
uint16x8_t test_vcgeq_f16(float16x8_t a, float16x8_t b) {
  return vcgeq_f16(a, b);
}

// CHECK-LABEL: test_vcgt_f16
// CHECK:  [[TMP1:%.*]] = fcmp ogt <4 x half> %a, %b
// CHECK:  [[TMP2:%.*]] = sext <4 x i1> [[TMP1]] to <4 x i16>
// CHECK:  ret <4 x i16> [[TMP2]]
uint16x4_t test_vcgt_f16(float16x4_t a, float16x4_t b) {
  return vcgt_f16(a, b);
}

// CHECK-LABEL: test_vcgtq_f16
// CHECK:  [[TMP1:%.*]] = fcmp ogt <8 x half> %a, %b
// CHECK:  [[TMP2:%.*]] = sext <8 x i1> [[TMP1:%.*]] to <8 x i16>
// CHECK:  ret <8 x i16> [[TMP2]]
uint16x8_t test_vcgtq_f16(float16x8_t a, float16x8_t b) {
  return vcgtq_f16(a, b);
}

// CHECK-LABEL: test_vcle_f16
// CHECK:  [[TMP1:%.*]] = fcmp ole <4 x half> %a, %b
// CHECK:  [[TMP2:%.*]] = sext <4 x i1> [[TMP1]] to <4 x i16>
// CHECK:  ret <4 x i16> [[TMP2]]
uint16x4_t test_vcle_f16(float16x4_t a, float16x4_t b) {
  return vcle_f16(a, b);
}

// CHECK-LABEL: test_vcleq_f16
// CHECK:  [[TMP1:%.*]] = fcmp ole <8 x half> %a, %b
// CHECK:  [[TMP2:%.*]] = sext <8 x i1> [[TMP1:%.*]] to <8 x i16>
// CHECK:  ret <8 x i16> [[TMP2]]
uint16x8_t test_vcleq_f16(float16x8_t a, float16x8_t b) {
  return vcleq_f16(a, b);
}

// CHECK-LABEL: test_vclt_f16
// CHECK:  [[TMP1:%.*]] = fcmp olt <4 x half> %a, %b
// CHECK:  [[TMP2:%.*]] = sext <4 x i1> [[TMP1]] to <4 x i16>
// CHECK:  ret <4 x i16> [[TMP2]]
uint16x4_t test_vclt_f16(float16x4_t a, float16x4_t b) {
  return vclt_f16(a, b);
}

// CHECK-LABEL: test_vcltq_f16
// CHECK:  [[TMP1:%.*]] = fcmp olt <8 x half> %a, %b
// CHECK:  [[TMP2:%.*]] = sext <8 x i1> [[TMP1:%.*]] to <8 x i16>
// CHECK:  ret <8 x i16> [[TMP2]]
uint16x8_t test_vcltq_f16(float16x8_t a, float16x8_t b) {
  return vcltq_f16(a, b);
}

// CHECK-LABEL: test_vcvt_n_f16_s16
// CHECK:  [[CVT:%.*]] = call <4 x half> @llvm.aarch64.neon.vcvtfxs2fp.v4f16.v4i16(<4 x i16> %vcvt_n, i32 2)
// CHECK:  ret <4 x half> [[CVT]]
float16x4_t test_vcvt_n_f16_s16(int16x4_t a) {
  return vcvt_n_f16_s16(a, 2);
}

// CHECK-LABEL: test_vcvtq_n_f16_s16
// CHECK:  [[CVT:%.*]] = call <8 x half> @llvm.aarch64.neon.vcvtfxs2fp.v8f16.v8i16(<8 x i16> %vcvt_n, i32 2)
// CHECK:  ret <8 x half> [[CVT]]
float16x8_t test_vcvtq_n_f16_s16(int16x8_t a) {
  return vcvtq_n_f16_s16(a, 2);
}

// CHECK-LABEL: test_vcvt_n_f16_u16
// CHECK:  [[CVT:%.*]] = call <4 x half> @llvm.aarch64.neon.vcvtfxu2fp.v4f16.v4i16(<4 x i16> %vcvt_n, i32 2)
// CHECK:  ret <4 x half> [[CVT]]
float16x4_t test_vcvt_n_f16_u16(uint16x4_t a) {
  return vcvt_n_f16_u16(a, 2);
}

// CHECK-LABEL: test_vcvtq_n_f16_u16
// CHECK:  [[CVT:%.*]] = call <8 x half> @llvm.aarch64.neon.vcvtfxu2fp.v8f16.v8i16(<8 x i16> %vcvt_n, i32 2)
// CHECK:  ret <8 x half> [[CVT]]
float16x8_t test_vcvtq_n_f16_u16(uint16x8_t a) {
  return vcvtq_n_f16_u16(a, 2);
}

// CHECK-LABEL: test_vcvt_n_s16_f16
// CHECK:  [[CVT:%.*]] = call <4 x i16> @llvm.aarch64.neon.vcvtfp2fxs.v4i16.v4f16(<4 x half> %vcvt_n, i32 2)
// CHECK:  ret <4 x i16> [[CVT]]
int16x4_t test_vcvt_n_s16_f16(float16x4_t a) {
  return vcvt_n_s16_f16(a, 2);
}

// CHECK-LABEL: test_vcvtq_n_s16_f16
// CHECK:  [[CVT:%.*]] = call <8 x i16> @llvm.aarch64.neon.vcvtfp2fxs.v8i16.v8f16(<8 x half> %vcvt_n, i32 2)
// CHECK:  ret <8 x i16> [[CVT]]
int16x8_t test_vcvtq_n_s16_f16(float16x8_t a) {
  return vcvtq_n_s16_f16(a, 2);
}

// CHECK-LABEL: test_vcvt_n_u16_f16
// CHECK:  [[CVT:%.*]] = call <4 x i16> @llvm.aarch64.neon.vcvtfp2fxu.v4i16.v4f16(<4 x half> %vcvt_n, i32 2)
// CHECK:  ret <4 x i16> [[CVT]]
uint16x4_t test_vcvt_n_u16_f16(float16x4_t a) {
  return vcvt_n_u16_f16(a, 2);
}

// CHECK-LABEL: test_vcvtq_n_u16_f16
// CHECK:  [[CVT:%.*]] = call <8 x i16> @llvm.aarch64.neon.vcvtfp2fxu.v8i16.v8f16(<8 x half> %vcvt_n, i32 2)
// CHECK:  ret <8 x i16> [[CVT]]
uint16x8_t test_vcvtq_n_u16_f16(float16x8_t a) {
  return vcvtq_n_u16_f16(a, 2);
}

// CHECK-LABEL: test_vdiv_f16
// CHECK:  [[DIV:%.*]] = fdiv <4 x half> %a, %b
// CHECK:  ret <4 x half> [[DIV]]
float16x4_t test_vdiv_f16(float16x4_t a, float16x4_t b) {
  return vdiv_f16(a, b);
}

// CHECK-LABEL: test_vdivq_f16
// CHECK:  [[DIV:%.*]] = fdiv <8 x half> %a, %b
// CHECK:  ret <8 x half> [[DIV]]
float16x8_t test_vdivq_f16(float16x8_t a, float16x8_t b) {
  return vdivq_f16(a, b);
}

// CHECK-LABEL: test_vmax_f16
// CHECK:  [[MAX:%.*]] = call <4 x half> @llvm.aarch64.neon.fmax.v4f16(<4 x half> %a, <4 x half> %b)
// CHECK:  ret <4 x half> [[MAX]]
float16x4_t test_vmax_f16(float16x4_t a, float16x4_t b) {
  return vmax_f16(a, b);
}

// CHECK-LABEL: test_vmaxq_f16
// CHECK:  [[MAX:%.*]] = call <8 x half> @llvm.aarch64.neon.fmax.v8f16(<8 x half> %a, <8 x half> %b)
// CHECK:  ret <8 x half> [[MAX]]
float16x8_t test_vmaxq_f16(float16x8_t a, float16x8_t b) {
  return vmaxq_f16(a, b);
}

// CHECK-LABEL: test_vmaxnm_f16
// CHECK:  [[MAX:%.*]] = call <4 x half> @llvm.aarch64.neon.fmaxnm.v4f16(<4 x half> %a, <4 x half> %b)
// CHECK:  ret <4 x half> [[MAX]]
float16x4_t test_vmaxnm_f16(float16x4_t a, float16x4_t b) {
  return vmaxnm_f16(a, b);
}

// CHECK-LABEL: test_vmaxnmq_f16
// CHECK:  [[MAX:%.*]] = call <8 x half> @llvm.aarch64.neon.fmaxnm.v8f16(<8 x half> %a, <8 x half> %b)
// CHECK:  ret <8 x half> [[MAX]]
float16x8_t test_vmaxnmq_f16(float16x8_t a, float16x8_t b) {
  return vmaxnmq_f16(a, b);
}

// CHECK-LABEL: test_vmin_f16
// CHECK:  [[MIN:%.*]] = call <4 x half> @llvm.aarch64.neon.fmin.v4f16(<4 x half> %a, <4 x half> %b)
// CHECK:  ret <4 x half> [[MIN]]
float16x4_t test_vmin_f16(float16x4_t a, float16x4_t b) {
  return vmin_f16(a, b);
}

// CHECK-LABEL: test_vminq_f16
// CHECK:  [[MIN:%.*]] = call <8 x half> @llvm.aarch64.neon.fmin.v8f16(<8 x half> %a, <8 x half> %b)
// CHECK:  ret <8 x half> [[MIN]]
float16x8_t test_vminq_f16(float16x8_t a, float16x8_t b) {
  return vminq_f16(a, b);
}

// CHECK-LABEL: test_vminnm_f16
// CHECK:  [[MIN:%.*]] = call <4 x half> @llvm.aarch64.neon.fminnm.v4f16(<4 x half> %a, <4 x half> %b)
// CHECK:  ret <4 x half> [[MIN]]
float16x4_t test_vminnm_f16(float16x4_t a, float16x4_t b) {
  return vminnm_f16(a, b);
}

// CHECK-LABEL: test_vminnmq_f16
// CHECK:  [[MIN:%.*]] = call <8 x half> @llvm.aarch64.neon.fminnm.v8f16(<8 x half> %a, <8 x half> %b)
// CHECK:  ret <8 x half> [[MIN]]
float16x8_t test_vminnmq_f16(float16x8_t a, float16x8_t b) {
  return vminnmq_f16(a, b);
}

// CHECK-LABEL: test_vmul_f16
// CHECK:  [[MUL:%.*]] = fmul <4 x half> %a, %b
// CHECK:  ret <4 x half> [[MUL]]
float16x4_t test_vmul_f16(float16x4_t a, float16x4_t b) {
  return vmul_f16(a, b);
}

// CHECK-LABEL: test_vmulq_f16
// CHECK:  [[MUL:%.*]] = fmul <8 x half> %a, %b
// CHECK:  ret <8 x half> [[MUL]]
float16x8_t test_vmulq_f16(float16x8_t a, float16x8_t b) {
  return vmulq_f16(a, b);
}

// CHECK-LABEL: test_vmulx_f16
// CHECK:  [[MUL:%.*]] = call <4 x half> @llvm.aarch64.neon.fmulx.v4f16(<4 x half> %a, <4 x half> %b)
// CHECK:  ret <4 x half> [[MUL]]
float16x4_t test_vmulx_f16(float16x4_t a, float16x4_t b) {
  return vmulx_f16(a, b);
}

// CHECK-LABEL: test_vmulxq_f16
// CHECK:  [[MUL:%.*]] = call <8 x half> @llvm.aarch64.neon.fmulx.v8f16(<8 x half> %a, <8 x half> %b)
// CHECK:  ret <8 x half> [[MUL]]
float16x8_t test_vmulxq_f16(float16x8_t a, float16x8_t b) {
  return vmulxq_f16(a, b);
}

// CHECK-LABEL: test_vpadd_f16
// CHECK:  [[ADD:%.*]] = call <4 x half> @llvm.aarch64.neon.addp.v4f16(<4 x half> %a, <4 x half> %b)
// CHECK:  ret <4 x half> [[ADD]]
float16x4_t test_vpadd_f16(float16x4_t a, float16x4_t b) {
  return vpadd_f16(a, b);
}

// CHECK-LABEL: test_vpaddq_f16
// CHECK:  [[ADD:%.*]] = call <8 x half> @llvm.aarch64.neon.addp.v8f16(<8 x half> %a, <8 x half> %b)
// CHECK:  ret <8 x half> [[ADD]]
float16x8_t test_vpaddq_f16(float16x8_t a, float16x8_t b) {
  return vpaddq_f16(a, b);
}

// CHECK-LABEL: test_vpmax_f16
// CHECK:  [[MAX:%.*]] = call <4 x half> @llvm.aarch64.neon.fmaxp.v4f16(<4 x half> %a, <4 x half> %b)
// CHECK:  ret <4 x half> [[MAX]]
float16x4_t test_vpmax_f16(float16x4_t a, float16x4_t b) {
  return vpmax_f16(a, b);
}

// CHECK-LABEL: test_vpmaxq_f16
// CHECK:  [[MAX:%.*]] = call <8 x half> @llvm.aarch64.neon.fmaxp.v8f16(<8 x half> %a, <8 x half> %b)
// CHECK:  ret <8 x half> [[MAX]]
float16x8_t test_vpmaxq_f16(float16x8_t a, float16x8_t b) {
  return vpmaxq_f16(a, b);
}

// CHECK-LABEL: test_vpmaxnm_f16
// CHECK:  [[MAX:%.*]] = call <4 x half> @llvm.aarch64.neon.fmaxnmp.v4f16(<4 x half> %a, <4 x half> %b)
// CHECK:  ret <4 x half> [[MAX]]
float16x4_t test_vpmaxnm_f16(float16x4_t a, float16x4_t b) {
  return vpmaxnm_f16(a, b);
}

// CHECK-LABEL: test_vpmaxnmq_f16
// CHECK:  [[MAX:%.*]] = call <8 x half> @llvm.aarch64.neon.fmaxnmp.v8f16(<8 x half> %a, <8 x half> %b)
// CHECK:  ret <8 x half> [[MAX]]
float16x8_t test_vpmaxnmq_f16(float16x8_t a, float16x8_t b) {
  return vpmaxnmq_f16(a, b);
}

// CHECK-LABEL: test_vpmin_f16
// CHECK:  [[MIN:%.*]] = call <4 x half> @llvm.aarch64.neon.fminp.v4f16(<4 x half> %a, <4 x half> %b)
// CHECK:  ret <4 x half> [[MIN]]
float16x4_t test_vpmin_f16(float16x4_t a, float16x4_t b) {
  return vpmin_f16(a, b);
}

// CHECK-LABEL: test_vpminq_f16
// CHECK:  [[MIN:%.*]] = call <8 x half> @llvm.aarch64.neon.fminp.v8f16(<8 x half> %a, <8 x half> %b)
// CHECK:  ret <8 x half> [[MIN]]
float16x8_t test_vpminq_f16(float16x8_t a, float16x8_t b) {
  return vpminq_f16(a, b);
}

// CHECK-LABEL: test_vpminnm_f16
// CHECK:  [[MIN:%.*]] = call <4 x half> @llvm.aarch64.neon.fminnmp.v4f16(<4 x half> %a, <4 x half> %b)
// CHECK:  ret <4 x half> [[MIN]]
float16x4_t test_vpminnm_f16(float16x4_t a, float16x4_t b) {
  return vpminnm_f16(a, b);
}

// CHECK-LABEL: test_vpminnmq_f16
// CHECK:  [[MIN:%.*]] = call <8 x half> @llvm.aarch64.neon.fminnmp.v8f16(<8 x half> %a, <8 x half> %b)
// CHECK:  ret <8 x half> [[MIN]]
float16x8_t test_vpminnmq_f16(float16x8_t a, float16x8_t b) {
  return vpminnmq_f16(a, b);
}

// CHECK-LABEL: test_vrecps_f16
// CHECK:  [[MIN:%.*]] = call <4 x half> @llvm.aarch64.neon.frecps.v4f16(<4 x half> %a, <4 x half> %b)
// CHECK:  ret <4 x half> [[MIN]]
float16x4_t test_vrecps_f16(float16x4_t a, float16x4_t b) {
  return vrecps_f16(a, b);
}

// CHECK-LABEL: test_vrecpsq_f16
// CHECK:  [[MIN:%.*]] =  call <8 x half> @llvm.aarch64.neon.frecps.v8f16(<8 x half> %a, <8 x half> %b)
// CHECK:  ret <8 x half> [[MIN]]
float16x8_t test_vrecpsq_f16(float16x8_t a, float16x8_t b) {
  return vrecpsq_f16(a, b);
}

// CHECK-LABEL: test_vrsqrts_f16
// CHECK:  [[MIN:%.*]] = call <4 x half> @llvm.aarch64.neon.frsqrts.v4f16(<4 x half> %a, <4 x half> %b)
// CHECK:  ret <4 x half> [[MIN]]
float16x4_t test_vrsqrts_f16(float16x4_t a, float16x4_t b) {
  return vrsqrts_f16(a, b);
}

// CHECK-LABEL: test_vrsqrtsq_f16
// CHECK:  [[MIN:%.*]] =  call <8 x half> @llvm.aarch64.neon.frsqrts.v8f16(<8 x half> %a, <8 x half> %b)
// CHECK:  ret <8 x half> [[MIN]]
float16x8_t test_vrsqrtsq_f16(float16x8_t a, float16x8_t b) {
  return vrsqrtsq_f16(a, b);
}

// CHECK-LABEL: test_vsub_f16
// CHECK:  [[ADD:%.*]] = fsub <4 x half> %a, %b 
// CHECK:  ret <4 x half> [[ADD]]
float16x4_t test_vsub_f16(float16x4_t a, float16x4_t b) {
  return vsub_f16(a, b);
}

// CHECK-LABEL: test_vsubq_f16
// CHECK:  [[ADD:%.*]] = fsub <8 x half> %a, %b
// CHECK:  ret <8 x half> [[ADD]]
float16x8_t test_vsubq_f16(float16x8_t a, float16x8_t b) {
  return vsubq_f16(a, b);
}

// CHECK-LABEL: test_vfma_f16
// CHECK:  [[ADD:%.*]] = call <4 x half> @llvm.fma.v4f16(<4 x half> %b, <4 x half> %c, <4 x half> %a)
// CHECK:  ret <4 x half> [[ADD]]
float16x4_t test_vfma_f16(float16x4_t a, float16x4_t b, float16x4_t c) {
  return vfma_f16(a, b, c);
}

// CHECK-LABEL: test_vfmaq_f16
// CHECK:  [[ADD:%.*]] = call <8 x half> @llvm.fma.v8f16(<8 x half> %b, <8 x half> %c, <8 x half> %a)
// CHECK:  ret <8 x half> [[ADD]]
float16x8_t test_vfmaq_f16(float16x8_t a, float16x8_t b, float16x8_t c) {
  return vfmaq_f16(a, b, c);
}

// CHECK-LABEL: test_vfms_f16
// CHECK:  [[SUB:%.*]] = fsub <4 x half> <half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000>, %b
// CHECK:  [[ADD:%.*]] = call <4 x half> @llvm.fma.v4f16(<4 x half> [[SUB]], <4 x half> %c, <4 x half> %a)
// CHECK:  ret <4 x half> [[ADD]]
float16x4_t test_vfms_f16(float16x4_t a, float16x4_t b, float16x4_t c) {
  return vfms_f16(a, b, c);
}

// CHECK-LABEL: test_vfmsq_f16
// CHECK:  [[SUB:%.*]] = fsub <8 x half> <half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000>, %b
// CHECK:  [[ADD:%.*]] = call <8 x half> @llvm.fma.v8f16(<8 x half> [[SUB]], <8 x half> %c, <8 x half> %a)
// CHECK:  ret <8 x half> [[ADD]]
float16x8_t test_vfmsq_f16(float16x8_t a, float16x8_t b, float16x8_t c) {
  return vfmsq_f16(a, b, c);
}

// CHECK-LABEL: test_vfma_lane_f16
// CHECK: [[TMP0:%.*]] = bitcast <4 x half> %a to <8 x i8>
// CHECK: [[TMP1:%.*]] = bitcast <4 x half> %b to <8 x i8>
// CHECK: [[TMP2:%.*]] = bitcast <4 x half> %c to <8 x i8>
// CHECK: [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <4 x half>
// CHECK: [[LANE:%.*]] = shufflevector <4 x half> [[TMP3]], <4 x half> [[TMP3]], <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK: [[TMP4:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x half>
// CHECK: [[TMP5:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x half>
// CHECK: [[FMLA:%.*]] = call <4 x half> @llvm.fma.v4f16(<4 x half> [[TMP4]], <4 x half> [[LANE]], <4 x half> [[TMP5]])
// CHECK: ret <4 x half> [[FMLA]]
float16x4_t test_vfma_lane_f16(float16x4_t a, float16x4_t b, float16x4_t c) {
  return vfma_lane_f16(a, b, c, 3);
}

// CHECK-LABEL: test_vfmaq_lane_f16
// CHECK: [[TMP0:%.*]] = bitcast <8 x half> %a to <16 x i8>
// CHECK: [[TMP1:%.*]] = bitcast <8 x half> %b to <16 x i8>
// CHECK: [[TMP2:%.*]] = bitcast <4 x half> %c to <8 x i8>
// CHECK: [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <4 x half>
// CHECK: [[LANE:%.*]] = shufflevector <4 x half> [[TMP3]], <4 x half> [[TMP3]], <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
// CHECK: [[TMP4:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x half>
// CHECK: [[TMP5:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x half>
// CHECK: [[FMLA:%.*]] = call <8 x half> @llvm.fma.v8f16(<8 x half> [[TMP4]], <8 x half> [[LANE]], <8 x half> [[TMP5]])
// CHECK: ret <8 x half> [[FMLA]]
float16x8_t test_vfmaq_lane_f16(float16x8_t a, float16x8_t b, float16x4_t c) {
  return vfmaq_lane_f16(a, b, c, 3);
}

// CHECK-LABEL: test_vfma_laneq_f16
// CHECK: [[TMP0:%.*]] = bitcast <4 x half> %a to <8 x i8>
// CHECK: [[TMP1:%.*]] = bitcast <4 x half> %b to <8 x i8>
// CHECK: [[TMP2:%.*]] = bitcast <8 x half> %c to <16 x i8>
// CHECK: [[TMP3:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x half>
// CHECK: [[TMP4:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x half>
// CHECK: [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <8 x half>
// CHECK: [[LANE:%.*]] = shufflevector <8 x half> [[TMP5]], <8 x half> [[TMP5]], <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// CHECK: [[FMLA:%.*]] = call <4 x half> @llvm.fma.v4f16(<4 x half> [[LANE]], <4 x half> [[TMP4]], <4 x half> [[TMP3]])
// CHECK: ret <4 x half> [[FMLA]]
float16x4_t test_vfma_laneq_f16(float16x4_t a, float16x4_t b, float16x8_t c) {
  return vfma_laneq_f16(a, b, c, 7);
}

// CHECK-LABEL: test_vfmaq_laneq_f16
// CHECK: [[TMP0:%.*]] = bitcast <8 x half> %a to <16 x i8>
// CHECK: [[TMP1:%.*]] = bitcast <8 x half> %b to <16 x i8>
// CHECK: [[TMP2:%.*]] = bitcast <8 x half> %c to <16 x i8>
// CHECK: [[TMP3:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x half>
// CHECK: [[TMP4:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x half>
// CHECK: [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <8 x half>
// CHECK: [[LANE:%.*]] = shufflevector <8 x half> [[TMP5]], <8 x half> [[TMP5]], <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
// CHECK: [[FMLA:%.*]] = call <8 x half> @llvm.fma.v8f16(<8 x half> [[LANE]], <8 x half> [[TMP4]], <8 x half> [[TMP3]])
// CHECK: ret <8 x half> [[FMLA]]
float16x8_t test_vfmaq_laneq_f16(float16x8_t a, float16x8_t b, float16x8_t c) {
  return vfmaq_laneq_f16(a, b, c, 7);
}

// CHECK-LABEL: test_vfma_n_f16
// CHECK: [[TMP0:%.*]] = insertelement <4 x half> undef, half %c, i32 0
// CHECK: [[TMP1:%.*]] = insertelement <4 x half> [[TMP0]], half %c, i32 1
// CHECK: [[TMP2:%.*]] = insertelement <4 x half> [[TMP1]], half %c, i32 2
// CHECK: [[TMP3:%.*]] = insertelement <4 x half> [[TMP2]], half %c, i32 3
// CHECK: [[FMA:%.*]]  = call <4 x half> @llvm.fma.v4f16(<4 x half> %b, <4 x half> [[TMP3]], <4 x half> %a)
// CHECK: ret <4 x half> [[FMA]]
float16x4_t test_vfma_n_f16(float16x4_t a, float16x4_t b, float16_t c) {
  return vfma_n_f16(a, b, c);
}

// CHECK-LABEL: test_vfmaq_n_f16
// CHECK: [[TMP0:%.*]] = insertelement <8 x half> undef, half %c, i32 0
// CHECK: [[TMP1:%.*]] = insertelement <8 x half> [[TMP0]], half %c, i32 1
// CHECK: [[TMP2:%.*]] = insertelement <8 x half> [[TMP1]], half %c, i32 2
// CHECK: [[TMP3:%.*]] = insertelement <8 x half> [[TMP2]], half %c, i32 3
// CHECK: [[TMP4:%.*]] = insertelement <8 x half> [[TMP3]], half %c, i32 4
// CHECK: [[TMP5:%.*]] = insertelement <8 x half> [[TMP4]], half %c, i32 5
// CHECK: [[TMP6:%.*]] = insertelement <8 x half> [[TMP5]], half %c, i32 6
// CHECK: [[TMP7:%.*]] = insertelement <8 x half> [[TMP6]], half %c, i32 7
// CHECK: [[FMA:%.*]]  = call <8 x half> @llvm.fma.v8f16(<8 x half> %b, <8 x half> [[TMP7]], <8 x half> %a)
// CHECK: ret <8 x half> [[FMA]]
float16x8_t test_vfmaq_n_f16(float16x8_t a, float16x8_t b, float16_t c) {
  return vfmaq_n_f16(a, b, c);
}

// CHECK-LABEL: test_vfmah_lane_f16
// CHECK: [[TMP0:%.*]] = bitcast <4 x half> %c to <8 x i8>
// CHECK: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x half>
// CHECK: [[EXTR:%.*]] = extractelement <4 x half> [[TMP1]], i32 3
// CHECK: [[FMA:%.*]]  = call half @llvm.fma.f16(half %b, half [[EXTR]], half %a)
// CHECK: ret half [[FMA]]
float16_t test_vfmah_lane_f16(float16_t a, float16_t b, float16x4_t c) {
  return vfmah_lane_f16(a, b, c, 3);
}

// CHECK-LABEL: test_vfmah_laneq_f16
// CHECK: [[TMP0:%.*]] = bitcast <8 x half> %c to <16 x i8>
// CHECK: [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x half>
// CHECK: [[EXTR:%.*]] = extractelement <8 x half> [[TMP1]], i32 7
// CHECK: [[FMA:%.*]]  = call half @llvm.fma.f16(half %b, half [[EXTR]], half %a)
// CHECK: ret half [[FMA]]
float16_t test_vfmah_laneq_f16(float16_t a, float16_t b, float16x8_t c) {
  return vfmah_laneq_f16(a, b, c, 7);
}

// CHECK-LABEL: test_vfms_lane_f16
// CHECK: [[SUB:%.*]]  = fsub <4 x half> <half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000>, %b
// CHECK: [[TMP0:%.*]] = bitcast <4 x half> %a to <8 x i8>
// CHECK: [[TMP1:%.*]] = bitcast <4 x half> [[SUB]] to <8 x i8>
// CHECK: [[TMP2:%.*]] = bitcast <4 x half> %c to <8 x i8>
// CHECK: [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <4 x half>
// CHECK: [[LANE:%.*]] = shufflevector <4 x half> [[TMP3]], <4 x half> [[TMP3]], <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK: [[TMP4:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x half>
// CHECK: [[TMP5:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x half>
// CHECK: [[FMA:%.*]] = call <4 x half> @llvm.fma.v4f16(<4 x half> [[TMP4]], <4 x half> [[LANE]], <4 x half> [[TMP5]])
// CHECK: ret <4 x half> [[FMA]]
float16x4_t test_vfms_lane_f16(float16x4_t a, float16x4_t b, float16x4_t c) {
  return vfms_lane_f16(a, b, c, 3);
}

// CHECK-LABEL: test_vfmsq_lane_f16
// CHECK: [[SUB:%.*]]  = fsub <8 x half> <half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000>, %b
// CHECK: [[TMP0:%.*]] = bitcast <8 x half> %a to <16 x i8>
// CHECK: [[TMP1:%.*]] = bitcast <8 x half> [[SUB]] to <16 x i8>
// CHECK: [[TMP2:%.*]] = bitcast <4 x half> %c to <8 x i8>
// CHECK: [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <4 x half>
// CHECK: [[LANE:%.*]] = shufflevector <4 x half> [[TMP3]], <4 x half> [[TMP3]], <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
// CHECK: [[TMP4:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x half>
// CHECK: [[TMP5:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x half>
// CHECK: [[FMLA:%.*]] = call <8 x half> @llvm.fma.v8f16(<8 x half> [[TMP4]], <8 x half> [[LANE]], <8 x half> [[TMP5]])
// CHECK: ret <8 x half> [[FMLA]]
float16x8_t test_vfmsq_lane_f16(float16x8_t a, float16x8_t b, float16x4_t c) {
  return vfmsq_lane_f16(a, b, c, 3);
}

// CHECK-LABEL: test_vfms_laneq_f16
// CHECK: [[SUB:%.*]]  = fsub <4 x half> <half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000>, %b
// CHECK: [[TMP0:%.*]] = bitcast <4 x half> %a to <8 x i8>
// CHECK: [[TMP1:%.*]] = bitcast <4 x half> [[SUB]] to <8 x i8>
// CHECK: [[TMP2:%.*]] = bitcast <8 x half> %c to <16 x i8>
// CHECK: [[TMP3:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x half>
// CHECK: [[TMP4:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x half>
// CHECK: [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <8 x half>
// CHECK: [[LANE:%.*]] = shufflevector <8 x half> [[TMP5]], <8 x half> [[TMP5]], <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// CHECK: [[FMLA:%.*]] = call <4 x half> @llvm.fma.v4f16(<4 x half> [[LANE]], <4 x half> [[TMP4]], <4 x half> [[TMP3]])
// CHECK: ret <4 x half> [[FMLA]]
float16x4_t test_vfms_laneq_f16(float16x4_t a, float16x4_t b, float16x8_t c) {
  return vfms_laneq_f16(a, b, c, 7);
}

// CHECK-LABEL: test_vfmsq_laneq_f16
// CHECK: [[SUB:%.*]]  = fsub <8 x half> <half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000>, %b
// CHECK: [[TMP0:%.*]] = bitcast <8 x half> %a to <16 x i8>
// CHECK: [[TMP1:%.*]] = bitcast <8 x half> [[SUB]] to <16 x i8>
// CHECK: [[TMP2:%.*]] = bitcast <8 x half> %c to <16 x i8>
// CHECK: [[TMP3:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x half>
// CHECK: [[TMP4:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x half>
// CHECK: [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <8 x half>
// CHECK: [[LANE:%.*]] = shufflevector <8 x half> [[TMP5]], <8 x half> [[TMP5]], <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
// CHECK: [[FMLA:%.*]] = call <8 x half> @llvm.fma.v8f16(<8 x half> [[LANE]], <8 x half> [[TMP4]], <8 x half> [[TMP3]])
// CHECK: ret <8 x half> [[FMLA]]
float16x8_t test_vfmsq_laneq_f16(float16x8_t a, float16x8_t b, float16x8_t c) {
  return vfmsq_laneq_f16(a, b, c, 7);
}

// CHECK-LABEL: test_vfms_n_f16
// CHECK: [[SUB:%.*]]  = fsub <4 x half> <half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000>, %b
// CHECK: [[TMP0:%.*]] = insertelement <4 x half> undef, half %c, i32 0
// CHECK: [[TMP1:%.*]] = insertelement <4 x half> [[TMP0]], half %c, i32 1
// CHECK: [[TMP2:%.*]] = insertelement <4 x half> [[TMP1]], half %c, i32 2
// CHECK: [[TMP3:%.*]] = insertelement <4 x half> [[TMP2]], half %c, i32 3
// CHECK: [[FMA:%.*]]  = call <4 x half> @llvm.fma.v4f16(<4 x half> [[SUB]], <4 x half> [[TMP3]], <4 x half> %a)
// CHECK: ret <4 x half> [[FMA]]
float16x4_t test_vfms_n_f16(float16x4_t a, float16x4_t b, float16_t c) {
  return vfms_n_f16(a, b, c);
}

// CHECK-LABEL: test_vfmsq_n_f16
// CHECK: [[SUB:%.*]]  = fsub <8 x half> <half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000, half 0xH8000>, %b
// CHECK: [[TMP0:%.*]] = insertelement <8 x half> undef, half %c, i32 0
// CHECK: [[TMP1:%.*]] = insertelement <8 x half> [[TMP0]], half %c, i32 1
// CHECK: [[TMP2:%.*]] = insertelement <8 x half> [[TMP1]], half %c, i32 2
// CHECK: [[TMP3:%.*]] = insertelement <8 x half> [[TMP2]], half %c, i32 3
// CHECK: [[TMP4:%.*]] = insertelement <8 x half> [[TMP3]], half %c, i32 4
// CHECK: [[TMP5:%.*]] = insertelement <8 x half> [[TMP4]], half %c, i32 5
// CHECK: [[TMP6:%.*]] = insertelement <8 x half> [[TMP5]], half %c, i32 6
// CHECK: [[TMP7:%.*]] = insertelement <8 x half> [[TMP6]], half %c, i32 7
// CHECK: [[FMA:%.*]]  = call <8 x half> @llvm.fma.v8f16(<8 x half> [[SUB]], <8 x half> [[TMP7]], <8 x half> %a)
// CHECK: ret <8 x half> [[FMA]]
float16x8_t test_vfmsq_n_f16(float16x8_t a, float16x8_t b, float16_t c) {
  return vfmsq_n_f16(a, b, c);
}

// CHECK-LABEL: test_vfmsh_lane_f16
// CHECK: [[TMP0:%.*]] = fpext half %b to float
// CHECK: [[TMP1:%.*]] = fsub float -0.000000e+00, [[TMP0]]
// CHECK: [[SUB:%.*]]  = fptrunc float [[TMP1]] to half
// CHECK: [[TMP2:%.*]] = bitcast <4 x half> %c to <8 x i8>
// CHECK: [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <4 x half>
// CHECK: [[EXTR:%.*]] = extractelement <4 x half> [[TMP3]], i32 3
// CHECK: [[FMA:%.*]]  = call half @llvm.fma.f16(half [[SUB]], half [[EXTR]], half %a)
// CHECK: ret half [[FMA]]
float16_t test_vfmsh_lane_f16(float16_t a, float16_t b, float16x4_t c) {
  return vfmsh_lane_f16(a, b, c, 3);
}

// CHECK-LABEL: test_vfmsh_laneq_f16
// CHECK: [[TMP0:%.*]] = fpext half %b to float
// CHECK: [[TMP1:%.*]] = fsub float -0.000000e+00, [[TMP0]]
// CHECK: [[SUB:%.*]]  = fptrunc float [[TMP1]] to half
// CHECK: [[TMP2:%.*]] = bitcast <8 x half> %c to <16 x i8>
// CHECK: [[TMP3:%.*]] = bitcast <16 x i8> [[TMP2]] to <8 x half>
// CHECK: [[EXTR:%.*]] = extractelement <8 x half> [[TMP3]], i32 7
// CHECK: [[FMA:%.*]]  = call half @llvm.fma.f16(half [[SUB]], half [[EXTR]], half %a)
// CHECK: ret half [[FMA]]
float16_t test_vfmsh_laneq_f16(float16_t a, float16_t b, float16x8_t c) {
  return vfmsh_laneq_f16(a, b, c, 7);
}

// CHECK-LABEL: test_vmul_lane_f16
// CHECK: [[TMP0:%.*]] = shufflevector <4 x half> %b, <4 x half> %b, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK: [[MUL:%.*]]  = fmul <4 x half> %a, [[TMP0]]
// CHECK: ret <4 x half> [[MUL]]
float16x4_t test_vmul_lane_f16(float16x4_t a, float16x4_t b) {
  return vmul_lane_f16(a, b, 3);
}

// CHECK-LABEL: test_vmulq_lane_f16
// CHECK: [[TMP0:%.*]] = shufflevector <4 x half> %b, <4 x half> %b, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
// CHECK: [[MUL:%.*]]  = fmul <8 x half> %a, [[TMP0]]
// CHECK: ret <8 x half> [[MUL]]
float16x8_t test_vmulq_lane_f16(float16x8_t a, float16x4_t b) {
  return vmulq_lane_f16(a, b, 7);
}

// CHECK-LABEL: test_vmul_laneq_f16
// CHECK: [[TMP0:%.*]] = shufflevector <8 x half> %b, <8 x half> %b, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// CHECK: [[MUL:%.*]]  = fmul <4 x half> %a, [[TMP0]]
// CHECK: ret <4 x half> [[MUL]]
float16x4_t test_vmul_laneq_f16(float16x4_t a, float16x8_t b) {
  return vmul_laneq_f16(a, b, 7);
}

// CHECK-LABEL: test_vmulq_laneq_f16
// CHECK: [[TMP0:%.*]] = shufflevector <8 x half> %b, <8 x half> %b, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
// CHECK: [[MUL:%.*]]  = fmul <8 x half> %a, [[TMP0]]
// CHECK: ret <8 x half> [[MUL]]
float16x8_t test_vmulq_laneq_f16(float16x8_t a, float16x8_t b) {
  return vmulq_laneq_f16(a, b, 7);
}

// CHECK-LABEL: test_vmul_n_f16
// CHECK: [[TMP0:%.*]] = insertelement <4 x half> undef, half %b, i32 0
// CHECK: [[TMP1:%.*]] = insertelement <4 x half> [[TMP0]], half %b, i32 1
// CHECK: [[TMP2:%.*]] = insertelement <4 x half> [[TMP1]], half %b, i32 2
// CHECK: [[TMP3:%.*]] = insertelement <4 x half> [[TMP2]], half %b, i32 3
// CHECK: [[MUL:%.*]]  = fmul <4 x half> %a, [[TMP3]]
// CHECK: ret <4 x half> [[MUL]]
float16x4_t test_vmul_n_f16(float16x4_t a, float16_t b) {
  return vmul_n_f16(a, b);
}

// CHECK-LABEL: test_vmulq_n_f16
// CHECK: [[TMP0:%.*]] = insertelement <8 x half> undef, half %b, i32 0
// CHECK: [[TMP1:%.*]] = insertelement <8 x half> [[TMP0]], half %b, i32 1
// CHECK: [[TMP2:%.*]] = insertelement <8 x half> [[TMP1]], half %b, i32 2
// CHECK: [[TMP3:%.*]] = insertelement <8 x half> [[TMP2]], half %b, i32 3
// CHECK: [[TMP4:%.*]] = insertelement <8 x half> [[TMP3]], half %b, i32 4
// CHECK: [[TMP5:%.*]] = insertelement <8 x half> [[TMP4]], half %b, i32 5
// CHECK: [[TMP6:%.*]] = insertelement <8 x half> [[TMP5]], half %b, i32 6
// CHECK: [[TMP7:%.*]] = insertelement <8 x half> [[TMP6]], half %b, i32 7
// CHECK: [[MUL:%.*]]  = fmul <8 x half> %a, [[TMP7]]
// CHECK: ret <8 x half> [[MUL]]
float16x8_t test_vmulq_n_f16(float16x8_t a, float16_t b) {
  return vmulq_n_f16(a, b);
}

// FIXME: Fix it when fp16 non-storage-only type becomes available.
// CHECK-LABEL: test_vmulh_lane_f16
// CHECK: [[CONV0:%.*]] = fpext half %a to float
// CHECK: [[CONV1:%.*]] = fpext half %{{.*}} to float
// CHECK: [[MUL:%.*]]   = fmul float [[CONV0:%.*]], [[CONV0:%.*]]
// CHECK: [[CONV3:%.*]] = fptrunc float %mul to half
// CHECK: ret half [[CONV3:%.*]]
float16_t test_vmulh_lane_f16(float16_t a, float16x4_t b) {
  return vmulh_lane_f16(a, b, 3);
}

// CHECK-LABEL: test_vmulh_laneq_f16
// CHECK: [[CONV0:%.*]] = fpext half %a to float
// CHECK: [[CONV1:%.*]] = fpext half %{{.*}} to float
// CHECK: [[MUL:%.*]]   = fmul float [[CONV0:%.*]], [[CONV0:%.*]]
// CHECK: [[CONV3:%.*]] = fptrunc float %mul to half
// CHECK: ret half [[CONV3:%.*]]
float16_t test_vmulh_laneq_f16(float16_t a, float16x8_t b) {
  return vmulh_laneq_f16(a, b, 7);
}

// CHECK-LABEL: test_vmulx_lane_f16
// CHECK: [[TMP0:%.*]] = shufflevector <4 x half> %b, <4 x half> %b, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK: [[MUL:%.*]] = call <4 x half> @llvm.aarch64.neon.fmulx.v4f16(<4 x half> %a, <4 x half> [[TMP0]])
// CHECK: ret <4 x half> [[MUL]]
float16x4_t test_vmulx_lane_f16(float16x4_t a, float16x4_t b) {
  return vmulx_lane_f16(a, b, 3);
}

// CHECK-LABEL: test_vmulxq_lane_f16
// CHECK: [[TMP0:%.*]] = shufflevector <4 x half> %b, <4 x half> %b, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
// CHECK: [[MUL:%.*]] = call <8 x half> @llvm.aarch64.neon.fmulx.v8f16(<8 x half> %a, <8 x half> [[TMP0]])
// CHECK: ret <8 x half> [[MUL]]
float16x8_t test_vmulxq_lane_f16(float16x8_t a, float16x4_t b) {
  return vmulxq_lane_f16(a, b, 7);
}

// CHECK-LABEL: test_vmulx_laneq_f16
// CHECK: [[TMP0:%.*]] = shufflevector <8 x half> %b, <8 x half> %b, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// CHECK: [[MUL:%.*]]  = call <4 x half> @llvm.aarch64.neon.fmulx.v4f16(<4 x half> %a, <4 x half> [[TMP0]])
// CHECK: ret <4 x half> [[MUL]]
float16x4_t test_vmulx_laneq_f16(float16x4_t a, float16x8_t b) {
  return vmulx_laneq_f16(a, b, 7);
}

// CHECK-LABEL: test_vmulxq_laneq_f16
// CHECK: [[TMP0:%.*]] = shufflevector <8 x half> %b, <8 x half> %b, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
// CHECK: [[MUL:%.*]]  = call <8 x half> @llvm.aarch64.neon.fmulx.v8f16(<8 x half> %a, <8 x half> [[TMP0]])
// CHECK: ret <8 x half> [[MUL]]
float16x8_t test_vmulxq_laneq_f16(float16x8_t a, float16x8_t b) {
  return vmulxq_laneq_f16(a, b, 7);
}

// CHECK-LABEL: test_vmulx_n_f16
// CHECK: [[TMP0:%.*]] = insertelement <4 x half> undef, half %b, i32 0
// CHECK: [[TMP1:%.*]] = insertelement <4 x half> [[TMP0]], half %b, i32 1
// CHECK: [[TMP2:%.*]] = insertelement <4 x half> [[TMP1]], half %b, i32 2
// CHECK: [[TMP3:%.*]] = insertelement <4 x half> [[TMP2]], half %b, i32 3
// CHECK: [[MUL:%.*]]  = call <4 x half> @llvm.aarch64.neon.fmulx.v4f16(<4 x half> %a, <4 x half> [[TMP3]])
// CHECK: ret <4 x half> [[MUL]]
float16x4_t test_vmulx_n_f16(float16x4_t a, float16_t b) {
  return vmulx_n_f16(a, b);
}

// CHECK-LABEL: test_vmulxq_n_f16
// CHECK: [[TMP0:%.*]] = insertelement <8 x half> undef, half %b, i32 0
// CHECK: [[TMP1:%.*]] = insertelement <8 x half> [[TMP0]], half %b, i32 1
// CHECK: [[TMP2:%.*]] = insertelement <8 x half> [[TMP1]], half %b, i32 2
// CHECK: [[TMP3:%.*]] = insertelement <8 x half> [[TMP2]], half %b, i32 3
// CHECK: [[TMP4:%.*]] = insertelement <8 x half> [[TMP3]], half %b, i32 4
// CHECK: [[TMP5:%.*]] = insertelement <8 x half> [[TMP4]], half %b, i32 5
// CHECK: [[TMP6:%.*]] = insertelement <8 x half> [[TMP5]], half %b, i32 6
// CHECK: [[TMP7:%.*]] = insertelement <8 x half> [[TMP6]], half %b, i32 7
// CHECK: [[MUL:%.*]]  = call <8 x half> @llvm.aarch64.neon.fmulx.v8f16(<8 x half> %a, <8 x half> [[TMP7]])
// CHECK: ret <8 x half> [[MUL]]
float16x8_t test_vmulxq_n_f16(float16x8_t a, float16_t b) {
  return vmulxq_n_f16(a, b);
}

// CHECK-LABEL: test_vmulxh_lane_f16
// CHECK: [[TMP0:%.*]] = bitcast <4 x half> %b to <8 x i8>
// CHECK: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x half>
// CHECK: [[EXTR:%.*]] = extractelement <4 x half> [[TMP1]], i32 3
// CHECK: [[MULX:%.*]] = call half @llvm.aarch64.neon.fmulx.f16(half %a, half [[EXTR]]
// CHECK: ret half [[MULX]]
float16_t test_vmulxh_lane_f16(float16_t a, float16x4_t b) {
  return vmulxh_lane_f16(a, b, 3);
}

// CHECK-LABEL: test_vmulxh_laneq_f16
// CHECK: [[TMP0:%.*]] = bitcast <8 x half> %b to <16 x i8>
// CHECK: [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x half>
// CHECK: [[EXTR:%.*]] = extractelement <8 x half> [[TMP1]], i32 7
// CHECK: [[MULX:%.*]] = call half @llvm.aarch64.neon.fmulx.f16(half %a, half [[EXTR]])
// CHECK: ret half [[MULX]]
float16_t test_vmulxh_laneq_f16(float16_t a, float16x8_t b) {
  return vmulxh_laneq_f16(a, b, 7);
}

// CHECK-LABEL: test_vmaxv_f16
// CHECK: [[TMP0:%.*]] = bitcast <4 x half> %a to <8 x i8>
// CHECK: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x half>
// CHECK: [[MAX:%.*]]  = call half @llvm.aarch64.neon.fmaxv.f16.v4f16(<4 x half> [[TMP1]])
// CHECK: ret half [[MAX]]
float16_t test_vmaxv_f16(float16x4_t a) {
  return vmaxv_f16(a);
}

// CHECK-LABEL: test_vmaxvq_f16
// CHECK: [[TMP0:%.*]] = bitcast <8 x half> %a to <16 x i8>
// CHECK: [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x half>
// CHECK: [[MAX:%.*]]  = call half @llvm.aarch64.neon.fmaxv.f16.v8f16(<8 x half> [[TMP1]])
// CHECK: ret half [[MAX]]
float16_t test_vmaxvq_f16(float16x8_t a) {
  return vmaxvq_f16(a);
}

// CHECK-LABEL: test_vminv_f16
// CHECK: [[TMP0:%.*]] = bitcast <4 x half> %a to <8 x i8>
// CHECK: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x half>
// CHECK: [[MAX:%.*]]  = call half @llvm.aarch64.neon.fminv.f16.v4f16(<4 x half> [[TMP1]])
// CHECK: ret half [[MAX]]
float16_t test_vminv_f16(float16x4_t a) {
  return vminv_f16(a);
}

// CHECK-LABEL: test_vminvq_f16
// CHECK: [[TMP0:%.*]] = bitcast <8 x half> %a to <16 x i8>
// CHECK: [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x half>
// CHECK: [[MAX:%.*]]  = call half @llvm.aarch64.neon.fminv.f16.v8f16(<8 x half> [[TMP1]])
// CHECK: ret half [[MAX]]
float16_t test_vminvq_f16(float16x8_t a) {
  return vminvq_f16(a);
}

// CHECK-LABEL: test_vmaxnmv_f16
// CHECK: [[TMP0:%.*]] = bitcast <4 x half> %a to <8 x i8>
// CHECK: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x half>
// CHECK: [[MAX:%.*]]  = call half @llvm.aarch64.neon.fmaxnmv.f16.v4f16(<4 x half> [[TMP1]])
// CHECK: ret half [[MAX]]
float16_t test_vmaxnmv_f16(float16x4_t a) {
  return vmaxnmv_f16(a);
}

// CHECK-LABEL: test_vmaxnmvq_f16
// CHECK: [[TMP0:%.*]] = bitcast <8 x half> %a to <16 x i8>
// CHECK: [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x half>
// CHECK: [[MAX:%.*]]  = call half @llvm.aarch64.neon.fmaxnmv.f16.v8f16(<8 x half> [[TMP1]])
// CHECK: ret half [[MAX]]
float16_t test_vmaxnmvq_f16(float16x8_t a) {
  return vmaxnmvq_f16(a);
}

// CHECK-LABEL: test_vminnmv_f16
// CHECK: [[TMP0:%.*]] = bitcast <4 x half> %a to <8 x i8>
// CHECK: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x half>
// CHECK: [[MAX:%.*]]  = call half @llvm.aarch64.neon.fminnmv.f16.v4f16(<4 x half> [[TMP1]])
// CHECK: ret half [[MAX]]
float16_t test_vminnmv_f16(float16x4_t a) {
  return vminnmv_f16(a);
}

// CHECK-LABEL: test_vminnmvq_f16
// CHECK: [[TMP0:%.*]] = bitcast <8 x half> %a to <16 x i8>
// CHECK: [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x half>
// CHECK: [[MAX:%.*]]  = call half @llvm.aarch64.neon.fminnmv.f16.v8f16(<8 x half> [[TMP1]])
// CHECK: ret half [[MAX]]
float16_t test_vminnmvq_f16(float16x8_t a) {
  return vminnmvq_f16(a);
}

// CHECK-LABEL: test_vbsl_f16
// CHECK:  [[TMP0:%.*]] = bitcast <4 x half> %b to <8 x i8>
// CHECK:  [[TMP1:%.*]] = bitcast <4 x half> %c to <8 x i8>
// CHECK:  [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// CHECK:  [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// CHECK:  [[TMP4:%.*]] = and <4 x i16> %a, [[TMP2]]
// CHECK:  [[TMP5:%.*]] = xor <4 x i16> %a, <i16 -1, i16 -1, i16 -1, i16 -1>
// CHECK:  [[TMP6:%.*]] = and <4 x i16> [[TMP5]], [[TMP3]]
// CHECK:  [[TMP7:%.*]] = or <4 x i16> [[TMP4]], [[TMP6]]
// CHECK:  [[TMP8:%.*]] = bitcast <4 x i16> [[TMP7]] to <4 x half>
// CHECK:  ret <4 x half> [[TMP8]]
float16x4_t test_vbsl_f16(uint16x4_t a, float16x4_t b, float16x4_t c) {
  return vbsl_f16(a, b, c);
}

// CHECK-LABEL: test_vbslq_f16
// CHECK:  [[TMP0:%.*]] = bitcast <8 x half> %b to <16 x i8>
// CHECK:  [[TMP1:%.*]] = bitcast <8 x half> %c to <16 x i8>
// CHECK:  [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// CHECK:  [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// CHECK:  [[TMP4:%.*]] = and <8 x i16> %a, [[TMP2]]
// CHECK:  [[TMP5:%.*]] = xor <8 x i16> %a, <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
// CHECK:  [[TMP6:%.*]] = and <8 x i16> [[TMP5]], [[TMP3]]
// CHECK:  [[TMP7:%.*]] = or <8 x i16> [[TMP4]], [[TMP6]]
// CHECK:  [[TMP8:%.*]] = bitcast <8 x i16> [[TMP7]] to <8 x half>
// CHECK:  ret <8 x half> [[TMP8]]
float16x8_t test_vbslq_f16(uint16x8_t a, float16x8_t b, float16x8_t c) {
  return vbslq_f16(a, b, c);
}

// CHECK-LABEL: test_vzip_f16
// CHECK:   [[RETVAL:%.*]]  = alloca %struct.float16x4x2_t, align 8
// CHECK:   [[__RET_I:%.*]] = alloca %struct.float16x4x2_t, align 8
// CHECK:   [[TMP0:%.*]]  = bitcast %struct.float16x4x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP1:%.*]]  = bitcast i8* [[TMP0]] to <4 x half>*
// CHECK:   [[VZIP0_I:%.*]] = shufflevector <4 x half> %a, <4 x half> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// CHECK:   store <4 x half> [[VZIP0_I]], <4 x half>* [[TMP1]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <4 x half>, <4 x half>* [[TMP1]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <4 x half> %a, <4 x half> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// CHECK:   store <4 x half> [[VZIP1_I]], <4 x half>* [[TMP2]]
float16x4x2_t test_vzip_f16(float16x4_t a, float16x4_t b) {
  return vzip_f16(a, b);
}

// CHECK-LABEL: test_vzipq_f16
// CHECK:   [[RETVAL:%.*]]  = alloca %struct.float16x8x2_t, align 16
// CHECK:   [[__RET_I:%.*]] = alloca %struct.float16x8x2_t, align 16
// CHECK:   [[TMP0:%.*]]  = bitcast %struct.float16x8x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP1:%.*]]  = bitcast i8* [[TMP0]] to <8 x half>*
// CHECK:   [[VZIP0_I:%.*]] = shufflevector <8 x half> %a, <8 x half> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// CHECK:   store <8 x half> [[VZIP0_I]], <8 x half>* [[TMP1]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <8 x half>, <8 x half>* [[TMP1]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <8 x half> %a, <8 x half> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// CHECK:   store <8 x half> [[VZIP1_I]], <8 x half>* [[TMP2]]
float16x8x2_t test_vzipq_f16(float16x8_t a, float16x8_t b) {
  return vzipq_f16(a, b);
}

// CHECK-LABEL: test_vuzp_f16
// CHECK:   [[RETVAL:%.*]]  = alloca %struct.float16x4x2_t, align 8
// CHECK:   [[__RET_I:%.*]] = alloca %struct.float16x4x2_t, align 8
// CHECK:   [[TMP0:%.*]]  = bitcast %struct.float16x4x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP1:%.*]]  = bitcast i8* [[TMP0]] to <4 x half>*
// CHECK:   [[VZIP0_I:%.*]] = shufflevector <4 x half> %a, <4 x half> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// CHECK:   store <4 x half> [[VZIP0_I]], <4 x half>* [[TMP1]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <4 x half>, <4 x half>* [[TMP1]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <4 x half> %a, <4 x half> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// CHECK:   store <4 x half> [[VZIP1_I]], <4 x half>* [[TMP2]]
float16x4x2_t test_vuzp_f16(float16x4_t a, float16x4_t b) {
  return vuzp_f16(a, b);
}

// CHECK-LABEL: test_vuzpq_f16
// CHECK:   [[RETVAL:%.*]]  = alloca %struct.float16x8x2_t, align 16
// CHECK:   [[__RET_I:%.*]] = alloca %struct.float16x8x2_t, align 16
// CHECK:   [[TMP0:%.*]]  = bitcast %struct.float16x8x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP1:%.*]]  = bitcast i8* [[TMP0]] to <8 x half>*
// CHECK:   [[VZIP0_I:%.*]] = shufflevector <8 x half> %a, <8 x half> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// CHECK:   store <8 x half> [[VZIP0_I]], <8 x half>* [[TMP1]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <8 x half>, <8 x half>* [[TMP1]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <8 x half> %a, <8 x half> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// CHECK:   store <8 x half> [[VZIP1_I]], <8 x half>* [[TMP2]]
float16x8x2_t test_vuzpq_f16(float16x8_t a, float16x8_t b) {
  return vuzpq_f16(a, b);
}

// CHECK-LABEL: test_vtrn_f16
// CHECK:   [[RETVAL:%.*]]  = alloca %struct.float16x4x2_t, align 8
// CHECK:   [[__RET_I:%.*]] = alloca %struct.float16x4x2_t, align 8
// CHECK:   [[TMP0:%.*]]  = bitcast %struct.float16x4x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP1:%.*]]  = bitcast i8* [[TMP0]] to <4 x half>*
// CHECK:   [[VZIP0_I:%.*]] = shufflevector <4 x half> %a, <4 x half> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
// CHECK:   store <4 x half> [[VZIP0_I]], <4 x half>* [[TMP1]]
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <4 x half>, <4 x half>* [[TMP1]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <4 x half> %a, <4 x half> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
// CHECK:   store <4 x half> [[VZIP1_I]], <4 x half>* [[TMP2]]
float16x4x2_t test_vtrn_f16(float16x4_t a, float16x4_t b) {
  return vtrn_f16(a, b);
}

// CHECK-LABEL: test_vtrnq_f16
// CHECK:   [[RETVAL:%.*]]  = alloca %struct.float16x8x2_t, align 16
// CHECK:   [[__RET_I:%.*]] = alloca %struct.float16x8x2_t, align 16
// CHECK:   [[TMP0:%.*]]  = bitcast %struct.float16x8x2_t* [[RETVAL]] to i8*
// CHECK:   [[TMP1:%.*]]  = bitcast i8* [[TMP0]] to <8 x half>*
// CHECK:   [[VZIP0_I:%.*]] = shufflevector <8 x half> %a, <8 x half> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
// CHECK:   store <8 x half> [[VZIP0_I]], <8 x half>* [[TMP1]] 
// CHECK:   [[TMP2:%.*]] = getelementptr inbounds <8 x half>, <8 x half>* [[TMP1]], i32 1
// CHECK:   [[VZIP1_I:%.*]] = shufflevector <8 x half> %a, <8 x half> %b, <8 x i32>  <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
// CHECK:   store <8 x half> [[VZIP1_I]], <8 x half>* [[TMP2]]
float16x8x2_t test_vtrnq_f16(float16x8_t a, float16x8_t b) {
  return vtrnq_f16(a, b);
}

// CHECK-LABEL: test_vmov_n_f16
// CHECK:   [[TMP0:%.*]] = insertelement <4 x half> undef, half %a, i32 0
// CHECK:   [[TMP1:%.*]] = insertelement <4 x half> [[TMP0]], half %a, i32 1
// CHECK:   [[TMP2:%.*]] = insertelement <4 x half> [[TMP1]], half %a, i32 2
// CHECK:   [[TMP3:%.*]] = insertelement <4 x half> [[TMP2]], half %a, i32 3
// CHECK:   ret <4 x half> [[TMP3]]
float16x4_t test_vmov_n_f16(float16_t a) {
  return vmov_n_f16(a);
}

// CHECK-LABEL: test_vmovq_n_f16
// CHECK:   [[TMP0:%.*]] = insertelement <8 x half> undef, half %a, i32 0
// CHECK:   [[TMP1:%.*]] = insertelement <8 x half> [[TMP0]], half %a, i32 1
// CHECK:   [[TMP2:%.*]] = insertelement <8 x half> [[TMP1]], half %a, i32 2
// CHECK:   [[TMP3:%.*]] = insertelement <8 x half> [[TMP2]], half %a, i32 3
// CHECK:   [[TMP4:%.*]] = insertelement <8 x half> [[TMP3]], half %a, i32 4
// CHECK:   [[TMP5:%.*]] = insertelement <8 x half> [[TMP4]], half %a, i32 5
// CHECK:   [[TMP6:%.*]] = insertelement <8 x half> [[TMP5]], half %a, i32 6
// CHECK:   [[TMP7:%.*]] = insertelement <8 x half> [[TMP6]], half %a, i32 7
// CHECK:   ret <8 x half> [[TMP7]]
float16x8_t test_vmovq_n_f16(float16_t a) {
  return vmovq_n_f16(a);
}

// CHECK-LABEL: test_vdup_n_f16
// CHECK:   [[TMP0:%.*]] = insertelement <4 x half> undef, half %a, i32 0
// CHECK:   [[TMP1:%.*]] = insertelement <4 x half> [[TMP0]], half %a, i32 1
// CHECK:   [[TMP2:%.*]] = insertelement <4 x half> [[TMP1]], half %a, i32 2
// CHECK:   [[TMP3:%.*]] = insertelement <4 x half> [[TMP2]], half %a, i32 3
// CHECK:   ret <4 x half> [[TMP3]]
float16x4_t test_vdup_n_f16(float16_t a) {
  return vdup_n_f16(a);
}

// CHECK-LABEL: test_vdupq_n_f16
// CHECK:   [[TMP0:%.*]] = insertelement <8 x half> undef, half %a, i32 0
// CHECK:   [[TMP1:%.*]] = insertelement <8 x half> [[TMP0]], half %a, i32 1
// CHECK:   [[TMP2:%.*]] = insertelement <8 x half> [[TMP1]], half %a, i32 2
// CHECK:   [[TMP3:%.*]] = insertelement <8 x half> [[TMP2]], half %a, i32 3
// CHECK:   [[TMP4:%.*]] = insertelement <8 x half> [[TMP3]], half %a, i32 4
// CHECK:   [[TMP5:%.*]] = insertelement <8 x half> [[TMP4]], half %a, i32 5
// CHECK:   [[TMP6:%.*]] = insertelement <8 x half> [[TMP5]], half %a, i32 6
// CHECK:   [[TMP7:%.*]] = insertelement <8 x half> [[TMP6]], half %a, i32 7
// CHECK:   ret <8 x half> [[TMP7]]
float16x8_t test_vdupq_n_f16(float16_t a) {
  return vdupq_n_f16(a);
}

// CHECK-LABEL: test_vdup_lane_f16
// CHECK:   [[SHFL:%.*]] = shufflevector <4 x half> %a, <4 x half> %a, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   ret <4 x half> [[SHFL]]
float16x4_t test_vdup_lane_f16(float16x4_t a) {
  return vdup_lane_f16(a, 3);
}

// CHECK-LABEL: test_vdupq_lane_f16
// CHECK:   [[SHFL:%.*]] = shufflevector <4 x half> %a, <4 x half> %a, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
// CHECK:   ret <8 x half> [[SHFL]]
float16x8_t test_vdupq_lane_f16(float16x4_t a) {
  return vdupq_lane_f16(a, 7);
}

// CHECK-LABEL: @test_vext_f16(
// CHECK:   [[TMP0:%.*]] = bitcast <4 x half> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <4 x half> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x half>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x half>
// CHECK:   [[VEXT:%.*]] = shufflevector <4 x half> [[TMP2]], <4 x half> [[TMP3]], <4 x i32> <i32 2, i32 3, i32 4, i32 5>
// CHECK:   ret <4 x half> [[VEXT]]
float16x4_t test_vext_f16(float16x4_t a, float16x4_t b) {
  return vext_f16(a, b, 2);
}

// CHECK-LABEL: @test_vextq_f16(
// CHECK:   [[TMP0:%.*]] = bitcast <8 x half> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x half> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x half>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x half>
// CHECK:   [[VEXT:%.*]] = shufflevector <8 x half> [[TMP2]], <8 x half> [[TMP3]], <8 x i32> <i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12>
// CHECK:   ret <8 x half> [[VEXT]]
float16x8_t test_vextq_f16(float16x8_t a, float16x8_t b) {
  return vextq_f16(a, b, 5);
}

// CHECK-LABEL: @test_vrev64_f16(
// CHECK:   [[SHFL:%.*]] = shufflevector <4 x half> %a, <4 x half> %a, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
// CHECK:   ret <4 x half> [[SHFL]]
float16x4_t test_vrev64_f16(float16x4_t a) {
  return vrev64_f16(a);
}

// CHECK-LABEL: @test_vrev64q_f16(
// CHECK:   [[SHFL:%.*]] = shufflevector <8 x half> %a, <8 x half> %a, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
// CHECK:   ret <8 x half> [[SHFL]]
float16x8_t test_vrev64q_f16(float16x8_t a) {
  return vrev64q_f16(a);
}

// CHECK-LABEL: @test_vzip1_f16(
// CHECK:   [[SHFL:%.*]] = shufflevector <4 x half> %a, <4 x half> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// CHECK:   ret <4 x half> [[SHFL]]
float16x4_t test_vzip1_f16(float16x4_t a, float16x4_t b) {
  return vzip1_f16(a, b);
}

// CHECK-LABEL: @test_vzip1q_f16(
// CHECK:   [[SHFL:%.*]] = shufflevector <8 x half> %a, <8 x half> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// CHECK:   ret <8 x half> [[SHFL]]
float16x8_t test_vzip1q_f16(float16x8_t a, float16x8_t b) {
  return vzip1q_f16(a, b);
}

// CHECK-LABEL: @test_vzip2_f16(
// CHECK:   [[SHFL:%.*]] = shufflevector <4 x half> %a, <4 x half> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// CHECK:   ret <4 x half> [[SHFL]]
float16x4_t test_vzip2_f16(float16x4_t a, float16x4_t b) {
  return vzip2_f16(a, b);
}

// CHECK-LABEL: @test_vzip2q_f16(
// CHECK:   [[SHFL:%.*]] = shufflevector <8 x half> %a, <8 x half> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// CHECK:   ret <8 x half> [[SHFL]]
float16x8_t test_vzip2q_f16(float16x8_t a, float16x8_t b) {
  return vzip2q_f16(a, b);
}

// CHECK-LABEL: @test_vuzp1_f16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x half> %a, <4 x half> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// CHECK:   ret <4 x half> [[SHUFFLE_I]]
float16x4_t test_vuzp1_f16(float16x4_t a, float16x4_t b) {
  return vuzp1_f16(a, b);
}

// CHECK-LABEL: @test_vuzp1q_f16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x half> %a, <8 x half> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// CHECK:   ret <8 x half> [[SHUFFLE_I]]
float16x8_t test_vuzp1q_f16(float16x8_t a, float16x8_t b) {
  return vuzp1q_f16(a, b);
}

// CHECK-LABEL: @test_vuzp2_f16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x half> %a, <4 x half> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// CHECK:   ret <4 x half> [[SHUFFLE_I]]
float16x4_t test_vuzp2_f16(float16x4_t a, float16x4_t b) {
  return vuzp2_f16(a, b);
}

// CHECK-LABEL: @test_vuzp2q_f16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x half> %a, <8 x half> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// CHECK:   ret <8 x half> [[SHUFFLE_I]]
float16x8_t test_vuzp2q_f16(float16x8_t a, float16x8_t b) {
  return vuzp2q_f16(a, b);
}

// CHECK-LABEL: @test_vtrn1_f16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x half> %a, <4 x half> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
// CHECK:   ret <4 x half> [[SHUFFLE_I]]
float16x4_t test_vtrn1_f16(float16x4_t a, float16x4_t b) {
  return vtrn1_f16(a, b);
}

// CHECK-LABEL: @test_vtrn1q_f16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x half> %a, <8 x half> %b, <8 x i32>  <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
// CHECK:   ret <8 x half> [[SHUFFLE_I]]
float16x8_t test_vtrn1q_f16(float16x8_t a, float16x8_t b) {
  return vtrn1q_f16(a, b);
}

// CHECK-LABEL: @test_vtrn2_f16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <4 x half> %a, <4 x half> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
// CHECK:   ret <4 x half> [[SHUFFLE_I]]
float16x4_t test_vtrn2_f16(float16x4_t a, float16x4_t b) {
  return vtrn2_f16(a, b);
}

// CHECK-LABEL: @test_vtrn2q_f16(
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <8 x half> %a, <8 x half> %b, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
// CHECK:   ret <8 x half> [[SHUFFLE_I]]
float16x8_t test_vtrn2q_f16(float16x8_t a, float16x8_t b) {
  return vtrn2q_f16(a, b);
}

