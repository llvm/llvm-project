// RUN: %clang_cc1 -no-opaque-pointers -triple armv8.2a-linux-gnu -target-abi apcs-gnu -target-feature +neon -target-feature +fullfp16 \
// RUN: -S -disable-O0-optnone -emit-llvm -o - %s \
// RUN: | opt -S -mem2reg \
// RUN: | FileCheck %s

// REQUIRES: arm-registered-target

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
// CHECK:  [[VCVT:%.*]] = call <4 x i16> @llvm.arm.neon.vcvtas.v4i16.v4f16(<4 x half> %a)
// CHECK:  ret <4 x i16> [[VCVT]]
int16x4_t test_vcvta_s16_f16 (float16x4_t a) {
  return vcvta_s16_f16(a);
}

// CHECK-LABEL: test_vcvta_u16_f16
// CHECK:  [[VCVT:%.*]] = call <4 x i16> @llvm.arm.neon.vcvtau.v4i16.v4f16(<4 x half> %a)
// CHECK:  ret <4 x i16> [[VCVT]]
int16x4_t test_vcvta_u16_f16 (float16x4_t a) {
   return vcvta_u16_f16(a);
}

// CHECK-LABEL: test_vcvtaq_s16_f16
// CHECK:  [[VCVT:%.*]] = call <8 x i16> @llvm.arm.neon.vcvtas.v8i16.v8f16(<8 x half> %a)
// CHECK:  ret <8 x i16> [[VCVT]]
int16x8_t test_vcvtaq_s16_f16 (float16x8_t a) {
  return vcvtaq_s16_f16(a);
}

// CHECK-LABEL: test_vcvtm_s16_f16
// CHECK:  [[VCVT:%.*]] = call <4 x i16> @llvm.arm.neon.vcvtms.v4i16.v4f16(<4 x half> %a)
// CHECK:  ret <4 x i16> [[VCVT]]
int16x4_t test_vcvtm_s16_f16 (float16x4_t a) {
  return vcvtm_s16_f16(a);
}

// CHECK-LABEL: test_vcvtmq_s16_f16
// CHECK:  [[VCVT:%.*]] = call <8 x i16> @llvm.arm.neon.vcvtms.v8i16.v8f16(<8 x half> %a)
// CHECK:  ret <8 x i16> [[VCVT]]
int16x8_t test_vcvtmq_s16_f16 (float16x8_t a) {
  return vcvtmq_s16_f16(a);
}

// CHECK-LABEL: test_vcvtm_u16_f16
// CHECK:  [[VCVT:%.*]] = call <4 x i16> @llvm.arm.neon.vcvtmu.v4i16.v4f16(<4 x half> %a)
// CHECK:  ret <4 x i16> [[VCVT]]
uint16x4_t test_vcvtm_u16_f16 (float16x4_t a) {
  return vcvtm_u16_f16(a);
}

// CHECK-LABEL: test_vcvtmq_u16_f16
// CHECK:  [[VCVT:%.*]] = call <8 x i16> @llvm.arm.neon.vcvtmu.v8i16.v8f16(<8 x half> %a)
// CHECK:  ret <8 x i16> [[VCVT]]
uint16x8_t test_vcvtmq_u16_f16 (float16x8_t a) {
  return vcvtmq_u16_f16(a);
}

// CHECK-LABEL: test_vcvtn_s16_f16
// CHECK:  [[VCVT:%.*]] = call <4 x i16> @llvm.arm.neon.vcvtns.v4i16.v4f16(<4 x half> %a)
// CHECK:  ret <4 x i16> [[VCVT]]
int16x4_t test_vcvtn_s16_f16 (float16x4_t a) {
  return vcvtn_s16_f16(a);
}

// CHECK-LABEL: test_vcvtnq_s16_f16
// CHECK:  [[VCVT:%.*]] = call <8 x i16> @llvm.arm.neon.vcvtns.v8i16.v8f16(<8 x half> %a)
// CHECK:  ret <8 x i16> [[VCVT]]
int16x8_t test_vcvtnq_s16_f16 (float16x8_t a) {
  return vcvtnq_s16_f16(a);
}

// CHECK-LABEL: test_vcvtn_u16_f16
// CHECK:  [[VCVT:%.*]] = call <4 x i16> @llvm.arm.neon.vcvtnu.v4i16.v4f16(<4 x half> %a)
// CHECK:  ret <4 x i16> [[VCVT]]
uint16x4_t test_vcvtn_u16_f16 (float16x4_t a) {
  return vcvtn_u16_f16(a);
}

// CHECK-LABEL: test_vcvtnq_u16_f16
// CHECK:  [[VCVT:%.*]] = call <8 x i16> @llvm.arm.neon.vcvtnu.v8i16.v8f16(<8 x half> %a)
// CHECK:  ret <8 x i16> [[VCVT]]
uint16x8_t test_vcvtnq_u16_f16 (float16x8_t a) {
  return vcvtnq_u16_f16(a);
}

// CHECK-LABEL: test_vcvtp_s16_f16
// CHECK:  [[VCVT:%.*]] = call <4 x i16> @llvm.arm.neon.vcvtps.v4i16.v4f16(<4 x half> %a)
// CHECK:  ret <4 x i16> [[VCVT]]
int16x4_t test_vcvtp_s16_f16 (float16x4_t a) {
  return vcvtp_s16_f16(a);
}

// CHECK-LABEL: test_vcvtpq_s16_f16
// CHECK:  [[VCVT:%.*]] = call <8 x i16> @llvm.arm.neon.vcvtps.v8i16.v8f16(<8 x half> %a)
// CHECK:  ret <8 x i16> [[VCVT]]
int16x8_t test_vcvtpq_s16_f16 (float16x8_t a) {
  return vcvtpq_s16_f16(a);
}

// CHECK-LABEL: test_vcvtp_u16_f16
// CHECK:  [[VCVT:%.*]] = call <4 x i16> @llvm.arm.neon.vcvtpu.v4i16.v4f16(<4 x half> %a)
// CHECK:  ret <4 x i16> [[VCVT]]
uint16x4_t test_vcvtp_u16_f16 (float16x4_t a) {
  return vcvtp_u16_f16(a);
}

// CHECK-LABEL: test_vcvtpq_u16_f16
// CHECK:  [[VCVT:%.*]] = call <8 x i16> @llvm.arm.neon.vcvtpu.v8i16.v8f16(<8 x half> %a)
// CHECK:  ret <8 x i16> [[VCVT]]
uint16x8_t test_vcvtpq_u16_f16 (float16x8_t a) {
  return vcvtpq_u16_f16(a);
}

// FIXME: Fix the zero constant when fp16 non-storage-only type becomes available.
// CHECK-LABEL: test_vneg_f16
// CHECK:  [[NEG:%.*]] = fneg <4 x half> %a
// CHECK:  ret <4 x half> [[NEG]]
float16x4_t test_vneg_f16(float16x4_t a) {
  return vneg_f16(a);
}

// CHECK-LABEL: test_vnegq_f16
// CHECK:  [[NEG:%.*]] = fneg <8 x half> %a
// CHECK:  ret <8 x half> [[NEG]]
float16x8_t test_vnegq_f16(float16x8_t a) {
  return vnegq_f16(a);
}

// CHECK-LABEL: test_vrecpe_f16
// CHECK:  [[RCP:%.*]] = call <4 x half> @llvm.arm.neon.vrecpe.v4f16(<4 x half> %a)
// CHECK:  ret <4 x half> [[RCP]]
float16x4_t test_vrecpe_f16(float16x4_t a) {
  return vrecpe_f16(a);
}

// CHECK-LABEL: test_vrecpeq_f16
// CHECK:  [[RCP:%.*]] = call <8 x half> @llvm.arm.neon.vrecpe.v8f16(<8 x half> %a)
// CHECK:  ret <8 x half> [[RCP]]
float16x8_t test_vrecpeq_f16(float16x8_t a) {
  return vrecpeq_f16(a);
}

// CHECK-LABEL: test_vrnd_f16
// CHECK:  [[RND:%.*]] =  call <4 x half> @llvm.arm.neon.vrintz.v4f16(<4 x half> %a)
// CHECK:  ret <4 x half> [[RND]]
float16x4_t test_vrnd_f16(float16x4_t a) {
  return vrnd_f16(a);
}

// CHECK-LABEL: test_vrndq_f16
// CHECK:  [[RND:%.*]] =  call <8 x half> @llvm.arm.neon.vrintz.v8f16(<8 x half> %a)
// CHECK:  ret <8 x half> [[RND]]
float16x8_t test_vrndq_f16(float16x8_t a) {
  return vrndq_f16(a);
}

// CHECK-LABEL: test_vrnda_f16
// CHECK:  [[RND:%.*]] =  call <4 x half> @llvm.arm.neon.vrinta.v4f16(<4 x half> %a)
// CHECK:  ret <4 x half> [[RND]]
float16x4_t test_vrnda_f16(float16x4_t a) {
  return vrnda_f16(a);
}

// CHECK-LABEL: test_vrndaq_f16
// CHECK:  [[RND:%.*]] =  call <8 x half> @llvm.arm.neon.vrinta.v8f16(<8 x half> %a)
// CHECK:  ret <8 x half> [[RND]]
float16x8_t test_vrndaq_f16(float16x8_t a) {
  return vrndaq_f16(a);
}

// CHECK-LABEL: test_vrndm_f16
// CHECK:  [[RND:%.*]] =  call <4 x half> @llvm.arm.neon.vrintm.v4f16(<4 x half> %a)
// CHECK:  ret <4 x half> [[RND]]
float16x4_t test_vrndm_f16(float16x4_t a) {
  return vrndm_f16(a);
}

// CHECK-LABEL: test_vrndmq_f16
// CHECK:  [[RND:%.*]] =  call <8 x half> @llvm.arm.neon.vrintm.v8f16(<8 x half> %a)
// CHECK:  ret <8 x half> [[RND]]
float16x8_t test_vrndmq_f16(float16x8_t a) {
  return vrndmq_f16(a);
}

// CHECK-LABEL: test_vrndn_f16
// CHECK:  [[RND:%.*]] =  call <4 x half> @llvm.arm.neon.vrintn.v4f16(<4 x half> %a)
// CHECK:  ret <4 x half> [[RND]]
float16x4_t test_vrndn_f16(float16x4_t a) {
  return vrndn_f16(a);
}

// CHECK-LABEL: test_vrndnq_f16
// CHECK:  [[RND:%.*]] =  call <8 x half> @llvm.arm.neon.vrintn.v8f16(<8 x half> %a)
// CHECK:  ret <8 x half> [[RND]]
float16x8_t test_vrndnq_f16(float16x8_t a) {
  return vrndnq_f16(a);
}

// CHECK-LABEL: test_vrndp_f16
// CHECK:  [[RND:%.*]] =  call <4 x half> @llvm.arm.neon.vrintp.v4f16(<4 x half> %a)
// CHECK:  ret <4 x half> [[RND]]
float16x4_t test_vrndp_f16(float16x4_t a) {
  return vrndp_f16(a);
}

// CHECK-LABEL: test_vrndpq_f16
// CHECK:  [[RND:%.*]] =  call <8 x half> @llvm.arm.neon.vrintp.v8f16(<8 x half> %a)
// CHECK:  ret <8 x half> [[RND]]
float16x8_t test_vrndpq_f16(float16x8_t a) {
  return vrndpq_f16(a);
}

// CHECK-LABEL: test_vrndx_f16
// CHECK:  [[RND:%.*]] =  call <4 x half> @llvm.arm.neon.vrintx.v4f16(<4 x half> %a)
// CHECK:  ret <4 x half> [[RND]]
float16x4_t test_vrndx_f16(float16x4_t a) {
  return vrndx_f16(a);
}

// CHECK-LABEL: test_vrndxq_f16
// CHECK:  [[RND:%.*]] =  call <8 x half> @llvm.arm.neon.vrintx.v8f16(<8 x half> %a)
// CHECK:  ret <8 x half> [[RND]]
float16x8_t test_vrndxq_f16(float16x8_t a) {
  return vrndxq_f16(a);
}

// CHECK-LABEL: test_vrsqrte_f16
// CHECK:  [[RND:%.*]] = call <4 x half> @llvm.arm.neon.vrsqrte.v4f16(<4 x half> %a)
// CHECK:  ret <4 x half> [[RND]]
float16x4_t test_vrsqrte_f16(float16x4_t a) {
  return vrsqrte_f16(a);
}

// CHECK-LABEL: test_vrsqrteq_f16
// CHECK:  [[RND:%.*]] = call <8 x half> @llvm.arm.neon.vrsqrte.v8f16(<8 x half> %a)
// CHECK:  ret <8 x half> [[RND]]
float16x8_t test_vrsqrteq_f16(float16x8_t a) {
  return vrsqrteq_f16(a);
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
// CHECK:  [[ABD:%.*]] = call <4 x half> @llvm.arm.neon.vabds.v4f16(<4 x half> %a, <4 x half> %b)
// CHECK:  ret <4 x half> [[ABD]]
float16x4_t test_vabd_f16(float16x4_t a, float16x4_t b) {
  return vabd_f16(a, b);
}

// CHECK-LABEL: test_vabdq_f16
// CHECK:  [[ABD:%.*]] = call <8 x half> @llvm.arm.neon.vabds.v8f16(<8 x half> %a, <8 x half> %b)
// CHECK:  ret <8 x half> [[ABD]]
float16x8_t test_vabdq_f16(float16x8_t a, float16x8_t b) {
  return vabdq_f16(a, b);
}

// CHECK-LABEL: test_vcage_f16
// CHECK:  [[ABS:%.*]] = call <4 x i16> @llvm.arm.neon.vacge.v4i16.v4f16(<4 x half> %a, <4 x half> %b)
// CHECK:  ret <4 x i16> [[ABS]]
uint16x4_t test_vcage_f16(float16x4_t a, float16x4_t b) {
  return vcage_f16(a, b);
}

// CHECK-LABEL: test_vcageq_f16
// CHECK:  [[ABS:%.*]] = call <8 x i16> @llvm.arm.neon.vacge.v8i16.v8f16(<8 x half> %a, <8 x half> %b)
// CHECK:  ret <8 x i16> [[ABS]]
uint16x8_t test_vcageq_f16(float16x8_t a, float16x8_t b) {
  return vcageq_f16(a, b);
}

// CHECK-LABEL: test_vcagt_f16
// CHECK:  [[ABS:%.*]] = call <4 x i16> @llvm.arm.neon.vacgt.v4i16.v4f16(<4 x half> %a, <4 x half> %b)
// CHECK:  ret <4 x i16> [[ABS]]
uint16x4_t test_vcagt_f16(float16x4_t a, float16x4_t b) {
  return vcagt_f16(a, b);
}

// CHECK-LABEL: test_vcagtq_f16
// CHECK:  [[ABS:%.*]] = call <8 x i16> @llvm.arm.neon.vacgt.v8i16.v8f16(<8 x half> %a, <8 x half> %b)
// CHECK:  ret <8 x i16> [[ABS]]
uint16x8_t test_vcagtq_f16(float16x8_t a, float16x8_t b) {
  return vcagtq_f16(a, b);
}

// CHECK-LABEL: test_vcale_f16
// CHECK:  [[ABS:%.*]] = call <4 x i16> @llvm.arm.neon.vacge.v4i16.v4f16(<4 x half> %b, <4 x half> %a)
// CHECK:  ret <4 x i16> [[ABS]]
uint16x4_t test_vcale_f16(float16x4_t a, float16x4_t b) {
  return vcale_f16(a, b);
}

// CHECK-LABEL: test_vcaleq_f16
// CHECK:  [[ABS:%.*]] = call <8 x i16> @llvm.arm.neon.vacge.v8i16.v8f16(<8 x half> %b, <8 x half> %a)
// CHECK:  ret <8 x i16> [[ABS]]
uint16x8_t test_vcaleq_f16(float16x8_t a, float16x8_t b) {
  return vcaleq_f16(a, b);
}

// CHECK-LABEL: test_vcalt_f16
// CHECK:  [[ABS:%.*]] = call <4 x i16> @llvm.arm.neon.vacgt.v4i16.v4f16(<4 x half> %b, <4 x half> %a)
// CHECK:  ret <4 x i16> [[ABS]]
uint16x4_t test_vcalt_f16(float16x4_t a, float16x4_t b) {
  return vcalt_f16(a, b);
}

// CHECK-LABEL: test_vcaltq_f16
// CHECK:  [[ABS:%.*]] = call <8 x i16> @llvm.arm.neon.vacgt.v8i16.v8f16(<8 x half> %b, <8 x half> %a)
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
// CHECK:  [[CVT:%.*]] = call <4 x half> @llvm.arm.neon.vcvtfxs2fp.v4f16.v4i16(<4 x i16> %vcvt_n, i32 2)
// CHECK:  ret <4 x half> [[CVT]]
float16x4_t test_vcvt_n_f16_s16(int16x4_t a) {
  return vcvt_n_f16_s16(a, 2);
}

// CHECK-LABEL: test_vcvtq_n_f16_s16
// CHECK:  [[CVT:%.*]] = call <8 x half> @llvm.arm.neon.vcvtfxs2fp.v8f16.v8i16(<8 x i16> %vcvt_n, i32 2)
// CHECK:  ret <8 x half> [[CVT]]
float16x8_t test_vcvtq_n_f16_s16(int16x8_t a) {
  return vcvtq_n_f16_s16(a, 2);
}

// CHECK-LABEL: test_vcvt_n_f16_u16
// CHECK:  [[CVT:%.*]] = call <4 x half> @llvm.arm.neon.vcvtfxu2fp.v4f16.v4i16(<4 x i16> %vcvt_n, i32 2)
// CHECK:  ret <4 x half> [[CVT]]
float16x4_t test_vcvt_n_f16_u16(uint16x4_t a) {
  return vcvt_n_f16_u16(a, 2);
}

// CHECK-LABEL: test_vcvtq_n_f16_u16
// CHECK:  [[CVT:%.*]] = call <8 x half> @llvm.arm.neon.vcvtfxu2fp.v8f16.v8i16(<8 x i16> %vcvt_n, i32 2)
// CHECK:  ret <8 x half> [[CVT]]
float16x8_t test_vcvtq_n_f16_u16(uint16x8_t a) {
  return vcvtq_n_f16_u16(a, 2);
}

// CHECK-LABEL: test_vcvt_n_s16_f16
// CHECK:  [[CVT:%.*]] = call <4 x i16> @llvm.arm.neon.vcvtfp2fxs.v4i16.v4f16(<4 x half> %vcvt_n, i32 2)
// CHECK:  ret <4 x i16> [[CVT]]
int16x4_t test_vcvt_n_s16_f16(float16x4_t a) {
  return vcvt_n_s16_f16(a, 2);
}

// CHECK-LABEL: test_vcvtq_n_s16_f16
// CHECK:  [[CVT:%.*]] = call <8 x i16> @llvm.arm.neon.vcvtfp2fxs.v8i16.v8f16(<8 x half> %vcvt_n, i32 2)
// CHECK:  ret <8 x i16> [[CVT]]
int16x8_t test_vcvtq_n_s16_f16(float16x8_t a) {
  return vcvtq_n_s16_f16(a, 2);
}

// CHECK-LABEL: test_vcvt_n_u16_f16
// CHECK:  [[CVT:%.*]] = call <4 x i16> @llvm.arm.neon.vcvtfp2fxu.v4i16.v4f16(<4 x half> %vcvt_n, i32 2)
// CHECK:  ret <4 x i16> [[CVT]]
uint16x4_t test_vcvt_n_u16_f16(float16x4_t a) {
  return vcvt_n_u16_f16(a, 2);
}

// CHECK-LABEL: test_vcvtq_n_u16_f16
// CHECK:  [[CVT:%.*]] = call <8 x i16> @llvm.arm.neon.vcvtfp2fxu.v8i16.v8f16(<8 x half> %vcvt_n, i32 2)
// CHECK:  ret <8 x i16> [[CVT]]
uint16x8_t test_vcvtq_n_u16_f16(float16x8_t a) {
  return vcvtq_n_u16_f16(a, 2);
}

// CHECK-LABEL: test_vmax_f16
// CHECK:  [[MAX:%.*]] = call <4 x half> @llvm.arm.neon.vmaxs.v4f16(<4 x half> %a, <4 x half> %b)
// CHECK:  ret <4 x half> [[MAX]]
float16x4_t test_vmax_f16(float16x4_t a, float16x4_t b) {
  return vmax_f16(a, b);
}

// CHECK-LABEL: test_vmaxq_f16
// CHECK:  [[MAX:%.*]] = call <8 x half> @llvm.arm.neon.vmaxs.v8f16(<8 x half> %a, <8 x half> %b)
// CHECK:  ret <8 x half> [[MAX]]
float16x8_t test_vmaxq_f16(float16x8_t a, float16x8_t b) {
  return vmaxq_f16(a, b);
}

// CHECK-LABEL: test_vmaxnm_f16
// CHECK:  [[MAX:%.*]] = call <4 x half> @llvm.arm.neon.vmaxnm.v4f16(<4 x half> %a, <4 x half> %b)
// CHECK:  ret <4 x half> [[MAX]]
float16x4_t test_vmaxnm_f16(float16x4_t a, float16x4_t b) {
  return vmaxnm_f16(a, b);
}

// CHECK-LABEL: test_vmaxnmq_f16
// CHECK:  [[MAX:%.*]] = call <8 x half> @llvm.arm.neon.vmaxnm.v8f16(<8 x half> %a, <8 x half> %b)
// CHECK:  ret <8 x half> [[MAX]]
float16x8_t test_vmaxnmq_f16(float16x8_t a, float16x8_t b) {
  return vmaxnmq_f16(a, b);
}

// CHECK-LABEL: test_vmin_f16
// CHECK:  [[MIN:%.*]] = call <4 x half> @llvm.arm.neon.vmins.v4f16(<4 x half> %a, <4 x half> %b)
// CHECK:  ret <4 x half> [[MIN]]
float16x4_t test_vmin_f16(float16x4_t a, float16x4_t b) {
  return vmin_f16(a, b);
}

// CHECK-LABEL: test_vminq_f16
// CHECK:  [[MIN:%.*]] = call <8 x half> @llvm.arm.neon.vmins.v8f16(<8 x half> %a, <8 x half> %b)
// CHECK:  ret <8 x half> [[MIN]]
float16x8_t test_vminq_f16(float16x8_t a, float16x8_t b) {
  return vminq_f16(a, b);
}

// CHECK-LABEL: test_vminnm_f16
// CHECK:  [[MIN:%.*]] = call <4 x half> @llvm.arm.neon.vminnm.v4f16(<4 x half> %a, <4 x half> %b)
// CHECK:  ret <4 x half> [[MIN]]
float16x4_t test_vminnm_f16(float16x4_t a, float16x4_t b) {
  return vminnm_f16(a, b);
}

// CHECK-LABEL: test_vminnmq_f16
// CHECK:  [[MIN:%.*]] = call <8 x half> @llvm.arm.neon.vminnm.v8f16(<8 x half> %a, <8 x half> %b)
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

// CHECK-LABEL: test_vpadd_f16
// CHECK:  [[ADD:%.*]] = call <4 x half> @llvm.arm.neon.vpadd.v4f16(<4 x half> %a, <4 x half> %b)
// CHECK:  ret <4 x half> [[ADD]]
float16x4_t test_vpadd_f16(float16x4_t a, float16x4_t b) {
  return vpadd_f16(a, b);
}

// CHECK-LABEL: test_vpmax_f16
// CHECK:  [[MAX:%.*]] = call <4 x half> @llvm.arm.neon.vpmaxs.v4f16(<4 x half> %a, <4 x half> %b)
// CHECK:  ret <4 x half> [[MAX]]
float16x4_t test_vpmax_f16(float16x4_t a, float16x4_t b) {
  return vpmax_f16(a, b);
}

// CHECK-LABEL: test_vpmin_f16
// CHECK:  [[MIN:%.*]] = call <4 x half> @llvm.arm.neon.vpmins.v4f16(<4 x half> %a, <4 x half> %b)
// CHECK:  ret <4 x half> [[MIN]]
float16x4_t test_vpmin_f16(float16x4_t a, float16x4_t b) {
  return vpmin_f16(a, b);
}

// CHECK-LABEL: test_vrecps_f16
// CHECK:  [[MIN:%.*]] = call <4 x half> @llvm.arm.neon.vrecps.v4f16(<4 x half> %a, <4 x half> %b)
// CHECK:  ret <4 x half> [[MIN]]
float16x4_t test_vrecps_f16(float16x4_t a, float16x4_t b) {
 return vrecps_f16(a, b);
}

// CHECK-LABEL: test_vrecpsq_f16
// CHECK:  [[MIN:%.*]] =  call <8 x half> @llvm.arm.neon.vrecps.v8f16(<8 x half> %a, <8 x half> %b)
// CHECK:  ret <8 x half> [[MIN]]
float16x8_t test_vrecpsq_f16(float16x8_t a, float16x8_t b) {
  return vrecpsq_f16(a, b);
}

// CHECK-LABEL: test_vrsqrts_f16
// CHECK:  [[MIN:%.*]] = call <4 x half> @llvm.arm.neon.vrsqrts.v4f16(<4 x half> %a, <4 x half> %b)
// CHECK:  ret <4 x half> [[MIN]]
float16x4_t test_vrsqrts_f16(float16x4_t a, float16x4_t b) {
  return vrsqrts_f16(a, b);
}

// CHECK-LABEL: test_vrsqrtsq_f16
// CHECK:  [[MIN:%.*]] =  call <8 x half> @llvm.arm.neon.vrsqrts.v8f16(<8 x half> %a, <8 x half> %b)
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
// CHECK:  [[SUB:%.*]] = fneg <4 x half> %b
// CHECK:  [[ADD:%.*]] = call <4 x half> @llvm.fma.v4f16(<4 x half> [[SUB]], <4 x half> %c, <4 x half> %a)
// CHECK:  ret <4 x half> [[ADD]]
float16x4_t test_vfms_f16(float16x4_t a, float16x4_t b, float16x4_t c) {
  return vfms_f16(a, b, c);
}

// CHECK-LABEL: test_vfmsq_f16
// CHECK:  [[SUB:%.*]] = fneg <8 x half> %b
// CHECK:  [[ADD:%.*]] = call <8 x half> @llvm.fma.v8f16(<8 x half> [[SUB]], <8 x half> %c, <8 x half> %a)
// CHECK:  ret <8 x half> [[ADD]]
float16x8_t test_vfmsq_f16(float16x8_t a, float16x8_t b, float16x8_t c) {
  return vfmsq_f16(a, b, c);
}

// CHECK-LABEL: test_vmul_lane_f16
// CHECK: [[TMP0:%.*]] = bitcast <4 x half> [[B:%.*]] to <8 x i8>
// CHECK: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x half>
// CHECK: [[LANE:%.*]] = shufflevector <4 x half> [[TMP1]], <4 x half> [[TMP1]], <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK: [[MUL:%.*]] = fmul <4 x half> [[A:%.*]], [[LANE]]
// CHECK: ret <4 x half> [[MUL]]
float16x4_t test_vmul_lane_f16(float16x4_t a, float16x4_t b) {
  return vmul_lane_f16(a, b, 3);
}

// CHECK-LABEL: test_vmulq_lane_f16
// CHECK: [[TMP0:%.*]] = bitcast <4 x half> [[B:%.*]] to <8 x i8>
// CHECK: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x half>
// CHECK: [[LANE:%.*]] = shufflevector <4 x half> [[TMP1]], <4 x half> [[TMP1]], <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
// CHECK: [[MUL:%.*]] = fmul <8 x half> [[A:%.*]], [[LANE]]
// CHECK: ret <8 x half> [[MUL]]
float16x8_t test_vmulq_lane_f16(float16x8_t a, float16x4_t b) {
  return vmulq_lane_f16(a, b, 3);
}

// CHECK-LABEL: test_vmul_n_f16
// CHECK: [[TMP0:%.*]] = insertelement <4 x half> undef, half [[b:%.*]], i32 0
// CHECK: [[TMP1:%.*]] = insertelement <4 x half> [[TMP0]], half [[b]], i32 1
// CHECK: [[TMP2:%.*]] = insertelement <4 x half> [[TMP1]], half [[b]], i32 2
// CHECK: [[TMP3:%.*]] = insertelement <4 x half> [[TMP2]], half [[b]], i32 3
// CHECK: [[MUL:%.*]]  = fmul <4 x half> %a, [[TMP3]]
// CHECK: ret <4 x half> [[MUL]]
float16x4_t test_vmul_n_f16(float16x4_t a, float16_t b) {
  return vmul_n_f16(a, b);
}

// CHECK-LABEL: test_vmulq_n_f16
// CHECK: [[TMP0:%.*]] = insertelement <8 x half> undef, half [[b:%.*]], i32 0
// CHECK: [[TMP1:%.*]] = insertelement <8 x half> [[TMP0]], half [[b]], i32 1
// CHECK: [[TMP2:%.*]] = insertelement <8 x half> [[TMP1]], half [[b]], i32 2
// CHECK: [[TMP3:%.*]] = insertelement <8 x half> [[TMP2]], half [[b]], i32 3
// CHECK: [[TMP4:%.*]] = insertelement <8 x half> [[TMP3]], half [[b]], i32 4
// CHECK: [[TMP5:%.*]] = insertelement <8 x half> [[TMP4]], half [[b]], i32 5
// CHECK: [[TMP6:%.*]] = insertelement <8 x half> [[TMP5]], half [[b]], i32 6
// CHECK: [[TMP7:%.*]] = insertelement <8 x half> [[TMP6]], half [[b]], i32 7
// CHECK: [[MUL:%.*]]  = fmul <8 x half> %a, [[TMP7]]
// CHECK: ret <8 x half> [[MUL]]
float16x8_t test_vmulq_n_f16(float16x8_t a, float16_t b) {
  return vmulq_n_f16(a, b);
}

// CHECK-LABEL: test_vbsl_f16
// CHECK:  [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// CHECK:  [[TMP1:%.*]] = bitcast <4 x half> %b to <8 x i8>
// CHECK:  [[TMP2:%.*]] = bitcast <4 x half> %c to <8 x i8>
// CHECK:  [[VBSL:%.*]] = call <8 x i8> @llvm.arm.neon.vbsl.v8i8(<8 x i8> [[TMP0]], <8 x i8> [[TMP1]], <8 x i8> [[TMP2]])
// CHECK:  [[TMP3:%.*]] = bitcast <8 x i8> [[VBSL]] to <4 x half>
// CHECK:  ret <4 x half> [[TMP3]]
float16x4_t test_vbsl_f16(uint16x4_t a, float16x4_t b, float16x4_t c) {
  return vbsl_f16(a, b, c);
}

// CHECK-LABEL: test_vbslq_f16
// CHECK:  [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// CHECK:  [[TMP1:%.*]] = bitcast <8 x half> %b to <16 x i8>
// CHECK:  [[TMP2:%.*]] = bitcast <8 x half> %c to <16 x i8>
// CHECK:  [[VBSL:%.*]] = call <16 x i8> @llvm.arm.neon.vbsl.v16i8(<16 x i8> [[TMP0]], <16 x i8> [[TMP1]], <16 x i8> [[TMP2]])
// CHECK:  [[TMP3:%.*]] = bitcast <16 x i8> [[VBSL]] to <8 x half>
// CHECK:  ret <8 x half> [[TMP3]]
float16x8_t test_vbslq_f16(uint16x8_t a, float16x8_t b, float16x8_t c) {
  return vbslq_f16(a, b, c);
}

// CHECK-LABEL: test_vzip_f16
// CHECK:  [[VZIP0:%.*]] = shufflevector <4 x half> %a, <4 x half> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
// CHECK:  store <4 x half> [[VZIP0]], <4 x half>* [[addr1:%.*]]
// CHECK:  [[VZIP1:%.*]] = shufflevector <4 x half> %a, <4 x half> %b, <4 x i32> <i32 2, i32 6, i32 3, i32 7>
// CHECK:  store <4 x half> [[VZIP1]], <4 x half>* [[addr2:%.*]]
float16x4x2_t test_vzip_f16(float16x4_t a, float16x4_t b) {
  return vzip_f16(a, b);
}

// CHECK-LABEL: test_vzipq_f16
// CHECK:  [[VZIP0:%.*]] = shufflevector <8 x half> %a, <8 x half> %b, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11>
// CHECK:  store <8 x half> [[VZIP0]], <8 x half>* [[addr1:%.*]]
// CHECK:  [[VZIP1:%.*]] = shufflevector <8 x half> %a, <8 x half> %b, <8 x i32> <i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
// CHECK:  store <8 x half> [[VZIP1]], <8 x half>* [[addr2:%.*]]
float16x8x2_t test_vzipq_f16(float16x8_t a, float16x8_t b) {
  return vzipq_f16(a, b);
}

// CHECK-LABEL: test_vuzp_f16
// CHECK:  [[VUZP0:%.*]] = shufflevector <4 x half> %a, <4 x half> %b, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// CHECK:  store <4 x half> [[VUZP0]], <4 x half>* [[addr1:%.*]]
// CHECK:  [[VUZP1:%.*]] = shufflevector <4 x half> %a, <4 x half> %b, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
// CHECK:  store <4 x half> [[VUZP1]], <4 x half>* [[addr1:%.*]]
float16x4x2_t test_vuzp_f16(float16x4_t a, float16x4_t b) {
  return vuzp_f16(a, b);
}

// CHECK-LABEL: test_vuzpq_f16
// CHECK:   [[VUZP0:%.*]] = shufflevector <8 x half> %a, <8 x half> %b, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
// CHECK:   store <8 x half> [[VUZP0]], <8 x half>* [[addr1:%.*]]
// CHECK:   [[VUZP1:%.*]] = shufflevector <8 x half> %a, <8 x half> %b, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
// CHECK:   store <8 x half> [[VUZP1]], <8 x half>* [[addr2:%.*]]
float16x8x2_t test_vuzpq_f16(float16x8_t a, float16x8_t b) {
  return vuzpq_f16(a, b);
}

// CHECK-LABEL: test_vtrn_f16
// CHECK:   [[VTRN0:%.*]] = shufflevector <4 x half> %a, <4 x half> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
// CHECK:   store <4 x half> [[VTRN0]], <4 x half>* [[addr1:%.*]]
// CHECK:   [[VTRN1:%.*]] = shufflevector <4 x half> %a, <4 x half> %b, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
// CHECK:   store <4 x half> [[VTRN1]], <4 x half>* [[addr2:%.*]]
float16x4x2_t test_vtrn_f16(float16x4_t a, float16x4_t b) {
  return vtrn_f16(a, b);
}

// CHECK-LABEL: test_vtrnq_f16
// CHECK:   [[VTRN0:%.*]] = shufflevector <8 x half> %a, <8 x half> %b, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
// CHECK:   store <8 x half> [[VTRN0]], <8 x half>* [[addr1:%.*]]
// CHECK:   [[VTRN1:%.*]] = shufflevector <8 x half> %a, <8 x half> %b, <8 x i32>  <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
// CHECK:   store <8 x half> [[VTRN1]], <8 x half>* [[addr2:%.*]]
float16x8x2_t test_vtrnq_f16(float16x8_t a, float16x8_t b) {
  return vtrnq_f16(a, b);
}

// CHECK-LABEL: test_vmov_n_f16
// CHECK:   [[TMP0:%.*]] = insertelement <4 x half> undef, half [[ARG:%.*]], i32 0
// CHECK:   [[TMP1:%.*]] = insertelement <4 x half> [[TMP0]], half [[ARG]], i32 1
// CHECK:   [[TMP2:%.*]] = insertelement <4 x half> [[TMP1]], half [[ARG]], i32 2
// CHECK:   [[TMP3:%.*]] = insertelement <4 x half> [[TMP2]], half [[ARG]], i32 3
// CHECK:   ret <4 x half> [[TMP3]]
float16x4_t test_vmov_n_f16(float16_t a) {
  return vmov_n_f16(a);
}

// CHECK-LABEL: test_vmovq_n_f16
// CHECK:   [[TMP0:%.*]] = insertelement <8 x half> undef, half [[ARG:%.*]], i32 0
// CHECK:   [[TMP1:%.*]] = insertelement <8 x half> [[TMP0]], half [[ARG]], i32 1
// CHECK:   [[TMP2:%.*]] = insertelement <8 x half> [[TMP1]], half [[ARG]], i32 2
// CHECK:   [[TMP3:%.*]] = insertelement <8 x half> [[TMP2]], half [[ARG]], i32 3
// CHECK:   [[TMP4:%.*]] = insertelement <8 x half> [[TMP3]], half [[ARG]], i32 4
// CHECK:   [[TMP5:%.*]] = insertelement <8 x half> [[TMP4]], half [[ARG]], i32 5
// CHECK:   [[TMP6:%.*]] = insertelement <8 x half> [[TMP5]], half [[ARG]], i32 6
// CHECK:   [[TMP7:%.*]] = insertelement <8 x half> [[TMP6]], half [[ARG]], i32 7
// CHECK:   ret <8 x half> [[TMP7]]
float16x8_t test_vmovq_n_f16(float16_t a) {
  return vmovq_n_f16(a);
}

// CHECK-LABEL: test_vdup_n_f16
// CHECK:   [[TMP0:%.*]] = insertelement <4 x half> undef, half [[ARG:%.*]], i32 0
// CHECK:   [[TMP1:%.*]] = insertelement <4 x half> [[TMP0]], half [[ARG]], i32 1
// CHECK:   [[TMP2:%.*]] = insertelement <4 x half> [[TMP1]], half [[ARG]], i32 2
// CHECK:   [[TMP3:%.*]] = insertelement <4 x half> [[TMP2]], half [[ARG]], i32 3
// CHECK:   ret <4 x half> [[TMP3]]
float16x4_t test_vdup_n_f16(float16_t a) {
  return vdup_n_f16(a);
}

// CHECK-LABEL: test_vdupq_n_f16
// CHECK:   [[TMP0:%.*]] = insertelement <8 x half> undef, half [[ARG:%.*]], i32 0
// CHECK:   [[TMP1:%.*]] = insertelement <8 x half> [[TMP0]], half [[ARG]], i32 1
// CHECK:   [[TMP2:%.*]] = insertelement <8 x half> [[TMP1]], half [[ARG]], i32 2
// CHECK:   [[TMP3:%.*]] = insertelement <8 x half> [[TMP2]], half [[ARG]], i32 3
// CHECK:   [[TMP4:%.*]] = insertelement <8 x half> [[TMP3]], half [[ARG]], i32 4
// CHECK:   [[TMP5:%.*]] = insertelement <8 x half> [[TMP4]], half [[ARG]], i32 5
// CHECK:   [[TMP6:%.*]] = insertelement <8 x half> [[TMP5]], half [[ARG]], i32 6
// CHECK:   [[TMP7:%.*]] = insertelement <8 x half> [[TMP6]], half [[ARG]], i32 7
// CHECK:   ret <8 x half> [[TMP7]]
float16x8_t test_vdupq_n_f16(float16_t a) {
  return vdupq_n_f16(a);
}

// CHECK-LABEL: test_vdup_lane_f16
// CHECK:   [[TMP0:%.*]] = bitcast <4 x half> [[A:%.*]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x half>
// CHECK:   [[LANE:%.*]] = shufflevector <4 x half> [[TMP1]], <4 x half> [[TMP1]], <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// CHECK:   ret <4 x half> [[LANE]]
float16x4_t test_vdup_lane_f16(float16x4_t a) {
  return vdup_lane_f16(a, 3);
}

// CHECK-LABEL: test_vdupq_lane_f16
// CHECK:   [[TMP0:%.*]] = bitcast <4 x half> [[A:%.*]] to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x half>
// CHECK:   [[LANE:%.*]] = shufflevector <4 x half> [[TMP1]], <4 x half> [[TMP1]], <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
// CHECK:   ret <8 x half> [[LANE]]
float16x8_t test_vdupq_lane_f16(float16x4_t a) {
  return vdupq_lane_f16(a, 3);
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
