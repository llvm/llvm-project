// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +fullfp16 -target-feature +v8.2a\
// RUN: -fallow-half-arguments-and-returns -flax-vector-conversions=none -S -disable-O0-optnone -emit-llvm -o - %s \
// RUN: | opt -S -mem2reg \
// RUN: | FileCheck --check-prefix=COMMON --check-prefix=COMMONIR --check-prefix=UNCONSTRAINED %s
// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +fullfp16 -target-feature +v8.2a\
// RUN: -ffp-exception-behavior=maytrap -DEXCEPT=1 \
// RUN: -fexperimental-strict-floating-point \
// RUN: -fallow-half-arguments-and-returns -flax-vector-conversions=none -S -disable-O0-optnone -emit-llvm -o - %s \
// RUN: | opt -S -mem2reg \
// RUN: | FileCheck --check-prefix=COMMON --check-prefix=COMMONIR --check-prefix=CONSTRAINED %s
// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +fullfp16 -target-feature +v8.2a\
// RUN: -fallow-half-arguments-and-returns -flax-vector-conversions=none -S -disable-O0-optnone -emit-llvm -o - %s \
// RUN: | opt -S -mem2reg | llc -o=- - \
// RUN: | FileCheck --check-prefix=COMMON --check-prefix=CHECK-ASM %s
// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-feature +fullfp16 -target-feature +v8.2a\
// RUN: -ffp-exception-behavior=maytrap -DEXCEPT=1 \
// RUN: -fexperimental-strict-floating-point \
// RUN: -fallow-half-arguments-and-returns -flax-vector-conversions=none -S -disable-O0-optnone -emit-llvm -o - %s \
// RUN: | opt -S -mem2reg | llc -o=- - \
// RUN: | FileCheck --check-prefix=COMMON --check-prefix=CHECK-ASM %s

// REQUIRES: aarch64-registered-target

// Test that the constrained intrinsics are picking up the exception
// metadata from the AST instead of the global default from the command line.
// FIXME: All cases of "fpexcept.maytrap" in this test are wrong.

#if EXCEPT
#pragma float_control(except, on)
#endif

#include <arm_neon.h>

// COMMON-LABEL: test_vsqrt_f16
// UNCONSTRAINED:  [[SQR:%.*]] = call <4 x half> @llvm.sqrt.v4f16(<4 x half> %a)
// CONSTRAINED:    [[SQR:%.*]] = call <4 x half> @llvm.experimental.constrained.sqrt.v4f16(<4 x half> %a, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// CHECK-ASM:      fsqrt v{{[0-9]+}}.4h, v{{[0-9]+}}.4h
// COMMONIR:       ret <4 x half> [[SQR]]
float16x4_t test_vsqrt_f16(float16x4_t a) {
  return vsqrt_f16(a);
}

// COMMON-LABEL: test_vsqrtq_f16
// UNCONSTRAINED:  [[SQR:%.*]] = call <8 x half> @llvm.sqrt.v8f16(<8 x half> %a)
// CONSTRAINED:    [[SQR:%.*]] = call <8 x half> @llvm.experimental.constrained.sqrt.v8f16(<8 x half> %a, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// CHECK-ASM:      fsqrt v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
// COMMONIR:       ret <8 x half> [[SQR]]
float16x8_t test_vsqrtq_f16(float16x8_t a) {
  return vsqrtq_f16(a);
}

// COMMON-LABEL: test_vfma_f16
// UNCONSTRAINED:  [[ADD:%.*]] = call <4 x half> @llvm.fma.v4f16(<4 x half> %b, <4 x half> %c, <4 x half> %a)
// CONSTRAINED:    [[ADD:%.*]] = call <4 x half> @llvm.experimental.constrained.fma.v4f16(<4 x half> %b, <4 x half> %c, <4 x half> %a, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// CHECK-ASM:      fmla v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h
// COMMONIR:       ret <4 x half> [[ADD]]
float16x4_t test_vfma_f16(float16x4_t a, float16x4_t b, float16x4_t c) {
  return vfma_f16(a, b, c);
}

// COMMON-LABEL: test_vfmaq_f16
// UNCONSTRAINED:  [[ADD:%.*]] = call <8 x half> @llvm.fma.v8f16(<8 x half> %b, <8 x half> %c, <8 x half> %a)
// CONSTRAINED:    [[ADD:%.*]] = call <8 x half> @llvm.experimental.constrained.fma.v8f16(<8 x half> %b, <8 x half> %c, <8 x half> %a, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// CHECK-ASM:      fmla v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
// COMMONIR:       ret <8 x half> [[ADD]]
float16x8_t test_vfmaq_f16(float16x8_t a, float16x8_t b, float16x8_t c) {
  return vfmaq_f16(a, b, c);
}

// COMMON-LABEL: test_vfms_f16
// COMMONIR:       [[SUB:%.*]] = fneg <4 x half> %b
// UNCONSTRAINED:  [[ADD:%.*]] = call <4 x half> @llvm.fma.v4f16(<4 x half> [[SUB]], <4 x half> %c, <4 x half> %a)
// CONSTRAINED:    [[ADD:%.*]] = call <4 x half> @llvm.experimental.constrained.fma.v4f16(<4 x half> [[SUB]], <4 x half> %c, <4 x half> %a, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// CHECK-ASM:      fmls v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.4h
// COMMONIR:       ret <4 x half> [[ADD]]
float16x4_t test_vfms_f16(float16x4_t a, float16x4_t b, float16x4_t c) {
  return vfms_f16(a, b, c);
}

// COMMON-LABEL: test_vfmsq_f16
// COMMONIR:       [[SUB:%.*]] = fneg <8 x half> %b
// UNCONSTRAINED:  [[ADD:%.*]] = call <8 x half> @llvm.fma.v8f16(<8 x half> [[SUB]], <8 x half> %c, <8 x half> %a)
// CONSTRAINED:    [[ADD:%.*]] = call <8 x half> @llvm.experimental.constrained.fma.v8f16(<8 x half> [[SUB]], <8 x half> %c, <8 x half> %a, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// CHECK-ASM:      fmls v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
// COMMONIR:       ret <8 x half> [[ADD]]
float16x8_t test_vfmsq_f16(float16x8_t a, float16x8_t b, float16x8_t c) {
  return vfmsq_f16(a, b, c);
}

// COMMON-LABEL: test_vfma_lane_f16
// COMMONIR:      [[TMP0:%.*]] = bitcast <4 x half> %a to <8 x i8>
// COMMONIR:      [[TMP1:%.*]] = bitcast <4 x half> %b to <8 x i8>
// COMMONIR:      [[TMP2:%.*]] = bitcast <4 x half> %c to <8 x i8>
// COMMONIR:      [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <4 x half>
// COMMONIR:      [[LANE:%.*]] = shufflevector <4 x half> [[TMP3]], <4 x half> [[TMP3]], <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// COMMONIR:      [[TMP4:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x half>
// COMMONIR:      [[TMP5:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x half>
// UNCONSTRAINED: [[FMLA:%.*]] = call <4 x half> @llvm.fma.v4f16(<4 x half> [[TMP4]], <4 x half> [[LANE]], <4 x half> [[TMP5]])
// CONSTRAINED:   [[FMLA:%.*]] = call <4 x half> @llvm.experimental.constrained.fma.v4f16(<4 x half> [[TMP4]], <4 x half> [[LANE]], <4 x half> [[TMP5]], metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// CHECK-ASM:     fmla v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.h[{{[0-9]+}}]
// COMMONIR:      ret <4 x half> [[FMLA]]
float16x4_t test_vfma_lane_f16(float16x4_t a, float16x4_t b, float16x4_t c) {
  return vfma_lane_f16(a, b, c, 3);
}

// COMMON-LABEL: test_vfmaq_lane_f16
// COMMONIR:      [[TMP0:%.*]] = bitcast <8 x half> %a to <16 x i8>
// COMMONIR:      [[TMP1:%.*]] = bitcast <8 x half> %b to <16 x i8>
// COMMONIR:      [[TMP2:%.*]] = bitcast <4 x half> %c to <8 x i8>
// COMMONIR:      [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <4 x half>
// COMMONIR:      [[LANE:%.*]] = shufflevector <4 x half> [[TMP3]], <4 x half> [[TMP3]], <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
// COMMONIR:      [[TMP4:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x half>
// COMMONIR:      [[TMP5:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x half>
// UNCONSTRAINED: [[FMLA:%.*]] = call <8 x half> @llvm.fma.v8f16(<8 x half> [[TMP4]], <8 x half> [[LANE]], <8 x half> [[TMP5]])
// CONSTRAINED:   [[FMLA:%.*]] = call <8 x half> @llvm.experimental.constrained.fma.v8f16(<8 x half> [[TMP4]], <8 x half> [[LANE]], <8 x half> [[TMP5]], metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// CHECK-ASM:     fmla v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.h[{{[0-9]+}}]
// COMMONIR:      ret <8 x half> [[FMLA]]
float16x8_t test_vfmaq_lane_f16(float16x8_t a, float16x8_t b, float16x4_t c) {
  return vfmaq_lane_f16(a, b, c, 3);
}

// COMMON-LABEL: test_vfma_laneq_f16
// COMMONIR:      [[TMP0:%.*]] = bitcast <4 x half> %a to <8 x i8>
// COMMONIR:      [[TMP1:%.*]] = bitcast <4 x half> %b to <8 x i8>
// COMMONIR:      [[TMP2:%.*]] = bitcast <8 x half> %c to <16 x i8>
// COMMONIR:      [[TMP3:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x half>
// COMMONIR:      [[TMP4:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x half>
// COMMONIR:      [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <8 x half>
// COMMONIR:      [[LANE:%.*]] = shufflevector <8 x half> [[TMP5]], <8 x half> [[TMP5]], <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// UNCONSTRAINED: [[FMLA:%.*]] = call <4 x half> @llvm.fma.v4f16(<4 x half> [[LANE]], <4 x half> [[TMP4]], <4 x half> [[TMP3]])
// CONSTRAINED:   [[FMLA:%.*]] = call <4 x half> @llvm.experimental.constrained.fma.v4f16(<4 x half> [[LANE]], <4 x half> [[TMP4]], <4 x half> [[TMP3]], metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// CHECK-ASM:     fmla v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.h[{{[0-9]+}}]
// COMMONIR:      ret <4 x half> [[FMLA]]
float16x4_t test_vfma_laneq_f16(float16x4_t a, float16x4_t b, float16x8_t c) {
  return vfma_laneq_f16(a, b, c, 7);
}

// COMMON-LABEL: test_vfmaq_laneq_f16
// COMMONIR:      [[TMP0:%.*]] = bitcast <8 x half> %a to <16 x i8>
// COMMONIR:      [[TMP1:%.*]] = bitcast <8 x half> %b to <16 x i8>
// COMMONIR:      [[TMP2:%.*]] = bitcast <8 x half> %c to <16 x i8>
// COMMONIR:      [[TMP3:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x half>
// COMMONIR:      [[TMP4:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x half>
// COMMONIR:      [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <8 x half>
// COMMONIR:      [[LANE:%.*]] = shufflevector <8 x half> [[TMP5]], <8 x half> [[TMP5]], <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
// UNCONSTRAINED: [[FMLA:%.*]] = call <8 x half> @llvm.fma.v8f16(<8 x half> [[LANE]], <8 x half> [[TMP4]], <8 x half> [[TMP3]])
// CONSTRAINED:   [[FMLA:%.*]] = call <8 x half> @llvm.experimental.constrained.fma.v8f16(<8 x half> [[LANE]], <8 x half> [[TMP4]], <8 x half> [[TMP3]], metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// CHECK-ASM:     fmla v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.h[{{[0-9]+}}]
// COMMONIR:      ret <8 x half> [[FMLA]]
float16x8_t test_vfmaq_laneq_f16(float16x8_t a, float16x8_t b, float16x8_t c) {
  return vfmaq_laneq_f16(a, b, c, 7);
}

// COMMON-LABEL: test_vfma_n_f16
// COMMONIR:      [[TMP0:%.*]] = insertelement <4 x half> undef, half %c, i32 0
// COMMONIR:      [[TMP1:%.*]] = insertelement <4 x half> [[TMP0]], half %c, i32 1
// COMMONIR:      [[TMP2:%.*]] = insertelement <4 x half> [[TMP1]], half %c, i32 2
// COMMONIR:      [[TMP3:%.*]] = insertelement <4 x half> [[TMP2]], half %c, i32 3
// UNCONSTRAINED: [[FMA:%.*]]  = call <4 x half> @llvm.fma.v4f16(<4 x half> %b, <4 x half> [[TMP3]], <4 x half> %a)
// CONSTRAINED:   [[FMA:%.*]]  = call <4 x half> @llvm.experimental.constrained.fma.v4f16(<4 x half> %b, <4 x half> [[TMP3]], <4 x half> %a, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// CHECK-ASM:     fmla v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.h[{{[0-9]+}}]
// COMMONIR:      ret <4 x half> [[FMA]]
float16x4_t test_vfma_n_f16(float16x4_t a, float16x4_t b, float16_t c) {
  return vfma_n_f16(a, b, c);
}

// COMMON-LABEL: test_vfmaq_n_f16
// COMMONIR:      [[TMP0:%.*]] = insertelement <8 x half> undef, half %c, i32 0
// COMMONIR:      [[TMP1:%.*]] = insertelement <8 x half> [[TMP0]], half %c, i32 1
// COMMONIR:      [[TMP2:%.*]] = insertelement <8 x half> [[TMP1]], half %c, i32 2
// COMMONIR:      [[TMP3:%.*]] = insertelement <8 x half> [[TMP2]], half %c, i32 3
// COMMONIR:      [[TMP4:%.*]] = insertelement <8 x half> [[TMP3]], half %c, i32 4
// COMMONIR:      [[TMP5:%.*]] = insertelement <8 x half> [[TMP4]], half %c, i32 5
// COMMONIR:      [[TMP6:%.*]] = insertelement <8 x half> [[TMP5]], half %c, i32 6
// COMMONIR:      [[TMP7:%.*]] = insertelement <8 x half> [[TMP6]], half %c, i32 7
// UNCONSTRAINED: [[FMA:%.*]]  = call <8 x half> @llvm.fma.v8f16(<8 x half> %b, <8 x half> [[TMP7]], <8 x half> %a)
// CONSTRAINED:   [[FMA:%.*]]  = call <8 x half> @llvm.experimental.constrained.fma.v8f16(<8 x half> %b, <8 x half> [[TMP7]], <8 x half> %a, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// CHECK-ASM:     fmla v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.h[{{[0-9]+}}]
// COMMONIR:      ret <8 x half> [[FMA]]
float16x8_t test_vfmaq_n_f16(float16x8_t a, float16x8_t b, float16_t c) {
  return vfmaq_n_f16(a, b, c);
}

// COMMON-LABEL: test_vfmah_lane_f16
// COMMONIR:      [[EXTR:%.*]] = extractelement <4 x half> %c, i32 3
// UNCONSTRAINED: [[FMA:%.*]]  = call half @llvm.fma.f16(half %b, half [[EXTR]], half %a)
// CONSTRAINED:   [[FMA:%.*]]  = call half @llvm.experimental.constrained.fma.f16(half %b, half [[EXTR]], half %a, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// CHECK-ASM:     fmla h{{[0-9]+}}, h{{[0-9]+}}, v{{[0-9]+}}.h[{{[0-9]+}}]
// COMMONIR:      ret half [[FMA]]
float16_t test_vfmah_lane_f16(float16_t a, float16_t b, float16x4_t c) {
  return vfmah_lane_f16(a, b, c, 3);
}

// COMMON-LABEL: test_vfmah_laneq_f16
// COMMONIR:      [[EXTR:%.*]] = extractelement <8 x half> %c, i32 7
// UNCONSTRAINED: [[FMA:%.*]]  = call half @llvm.fma.f16(half %b, half [[EXTR]], half %a)
// CONSTRAINED:   [[FMA:%.*]]  = call half @llvm.experimental.constrained.fma.f16(half %b, half [[EXTR]], half %a, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// CHECK-ASM:     fmla h{{[0-9]+}}, h{{[0-9]+}}, v{{[0-9]+}}.h[{{[0-9]+}}]
// COMMONIR:      ret half [[FMA]]
float16_t test_vfmah_laneq_f16(float16_t a, float16_t b, float16x8_t c) {
  return vfmah_laneq_f16(a, b, c, 7);
}

// COMMON-LABEL: test_vfms_lane_f16
// COMMONIR:      [[SUB:%.*]]  = fneg <4 x half> %b
// COMMONIR:      [[TMP0:%.*]] = bitcast <4 x half> %a to <8 x i8>
// COMMONIR:      [[TMP1:%.*]] = bitcast <4 x half> [[SUB]] to <8 x i8>
// COMMONIR:      [[TMP2:%.*]] = bitcast <4 x half> %c to <8 x i8>
// COMMONIR:      [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <4 x half>
// COMMONIR:      [[LANE:%.*]] = shufflevector <4 x half> [[TMP3]], <4 x half> [[TMP3]], <4 x i32> <i32 3, i32 3, i32 3, i32 3>
// COMMONIR:      [[TMP4:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x half>
// COMMONIR:      [[TMP5:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x half>
// UNCONSTRAINED: [[FMA:%.*]] = call <4 x half> @llvm.fma.v4f16(<4 x half> [[TMP4]], <4 x half> [[LANE]], <4 x half> [[TMP5]])
// CONSTRAINED:   [[FMA:%.*]] = call <4 x half> @llvm.experimental.constrained.fma.v4f16(<4 x half> [[TMP4]], <4 x half> [[LANE]], <4 x half> [[TMP5]], metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// CHECK-ASM:     fmls v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.h[{{[0-9]+}}]
// COMMONIR:      ret <4 x half> [[FMA]]
float16x4_t test_vfms_lane_f16(float16x4_t a, float16x4_t b, float16x4_t c) {
  return vfms_lane_f16(a, b, c, 3);
}

// COMMON-LABEL: test_vfmsq_lane_f16
// COMMONIR:      [[SUB:%.*]]  = fneg <8 x half> %b
// COMMONIR:      [[TMP0:%.*]] = bitcast <8 x half> %a to <16 x i8>
// COMMONIR:      [[TMP1:%.*]] = bitcast <8 x half> [[SUB]] to <16 x i8>
// COMMONIR:      [[TMP2:%.*]] = bitcast <4 x half> %c to <8 x i8>
// COMMONIR:      [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <4 x half>
// COMMONIR:      [[LANE:%.*]] = shufflevector <4 x half> [[TMP3]], <4 x half> [[TMP3]], <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
// COMMONIR:      [[TMP4:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x half>
// COMMONIR:      [[TMP5:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x half>
// UNCONSTRAINED: [[FMLA:%.*]] = call <8 x half> @llvm.fma.v8f16(<8 x half> [[TMP4]], <8 x half> [[LANE]], <8 x half> [[TMP5]])
// CONSTRAINED:   [[FMLA:%.*]] = call <8 x half> @llvm.experimental.constrained.fma.v8f16(<8 x half> [[TMP4]], <8 x half> [[LANE]], <8 x half> [[TMP5]], metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// CHECK-ASM:     fmls v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.h[{{[0-9]+}}]
// COMMONIR:      ret <8 x half> [[FMLA]]
float16x8_t test_vfmsq_lane_f16(float16x8_t a, float16x8_t b, float16x4_t c) {
  return vfmsq_lane_f16(a, b, c, 3);
}

// COMMON-LABEL: test_vfms_laneq_f16
// COMMONIR:      [[SUB:%.*]]  = fneg <4 x half> %b
// CHECK-ASM-NOT: fneg
// COMMONIR:      [[TMP0:%.*]] = bitcast <4 x half> %a to <8 x i8>
// COMMONIR:      [[TMP1:%.*]] = bitcast <4 x half> [[SUB]] to <8 x i8>
// COMMONIR:      [[TMP2:%.*]] = bitcast <8 x half> %c to <16 x i8>
// COMMONIR:      [[TMP3:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x half>
// COMMONIR:      [[TMP4:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x half>
// COMMONIR:      [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <8 x half>
// COMMONIR:      [[LANE:%.*]] = shufflevector <8 x half> [[TMP5]], <8 x half> [[TMP5]], <4 x i32> <i32 7, i32 7, i32 7, i32 7>
// UNCONSTRAINED: [[FMLA:%.*]] = call <4 x half> @llvm.fma.v4f16(<4 x half> [[LANE]], <4 x half> [[TMP4]], <4 x half> [[TMP3]])
// CONSTRAINED:   [[FMLA:%.*]] = call <4 x half> @llvm.experimental.constrained.fma.v4f16(<4 x half> [[LANE]], <4 x half> [[TMP4]], <4 x half> [[TMP3]], metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// CHECK-ASM:     fmls v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.h[{{[0-9]+}}]
// COMMONIR:      ret <4 x half> [[FMLA]]
float16x4_t test_vfms_laneq_f16(float16x4_t a, float16x4_t b, float16x8_t c) {
  return vfms_laneq_f16(a, b, c, 7);
}

// COMMON-LABEL: test_vfmsq_laneq_f16
// COMMONIR:      [[SUB:%.*]]  = fneg <8 x half> %b
// CHECK-ASM-NOT: fneg
// COMMONIR:      [[TMP0:%.*]] = bitcast <8 x half> %a to <16 x i8>
// COMMONIR:      [[TMP1:%.*]] = bitcast <8 x half> [[SUB]] to <16 x i8>
// COMMONIR:      [[TMP2:%.*]] = bitcast <8 x half> %c to <16 x i8>
// COMMONIR:      [[TMP3:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x half>
// COMMONIR:      [[TMP4:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x half>
// COMMONIR:      [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <8 x half>
// COMMONIR:      [[LANE:%.*]] = shufflevector <8 x half> [[TMP5]], <8 x half> [[TMP5]], <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
// UNCONSTRAINED: [[FMLA:%.*]] = call <8 x half> @llvm.fma.v8f16(<8 x half> [[LANE]], <8 x half> [[TMP4]], <8 x half> [[TMP3]])
// CONSTRAINED:   [[FMLA:%.*]] = call <8 x half> @llvm.experimental.constrained.fma.v8f16(<8 x half> [[LANE]], <8 x half> [[TMP4]], <8 x half> [[TMP3]], metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// CHECK-ASM:     fmls v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.h[{{[0-9]+}}]
// COMMONIR:      ret <8 x half> [[FMLA]]
float16x8_t test_vfmsq_laneq_f16(float16x8_t a, float16x8_t b, float16x8_t c) {
  return vfmsq_laneq_f16(a, b, c, 7);
}

// COMMON-LABEL: test_vfms_n_f16
// COMMONIR:      [[SUB:%.*]]  = fneg <4 x half> %b
// COMMONIR:      [[TMP0:%.*]] = insertelement <4 x half> undef, half %c, i32 0
// COMMONIR:      [[TMP1:%.*]] = insertelement <4 x half> [[TMP0]], half %c, i32 1
// COMMONIR:      [[TMP2:%.*]] = insertelement <4 x half> [[TMP1]], half %c, i32 2
// COMMONIR:      [[TMP3:%.*]] = insertelement <4 x half> [[TMP2]], half %c, i32 3
// UNCONSTRAINED: [[FMA:%.*]]  = call <4 x half> @llvm.fma.v4f16(<4 x half> [[SUB]], <4 x half> [[TMP3]], <4 x half> %a)
// CONSTRAINED:   [[FMA:%.*]]  = call <4 x half> @llvm.experimental.constrained.fma.v4f16(<4 x half> [[SUB]], <4 x half> [[TMP3]], <4 x half> %a, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// CHECK-ASM:     fmls v{{[0-9]+}}.4h, v{{[0-9]+}}.4h, v{{[0-9]+}}.h[{{[0-9]+}}]
// COMMONIR:      ret <4 x half> [[FMA]]
float16x4_t test_vfms_n_f16(float16x4_t a, float16x4_t b, float16_t c) {
  return vfms_n_f16(a, b, c);
}

// COMMON-LABEL: test_vfmsq_n_f16
// COMMONIR:      [[SUB:%.*]]  = fneg <8 x half> %b
// COMMONIR:      [[TMP0:%.*]] = insertelement <8 x half> undef, half %c, i32 0
// COMMONIR:      [[TMP1:%.*]] = insertelement <8 x half> [[TMP0]], half %c, i32 1
// COMMONIR:      [[TMP2:%.*]] = insertelement <8 x half> [[TMP1]], half %c, i32 2
// COMMONIR:      [[TMP3:%.*]] = insertelement <8 x half> [[TMP2]], half %c, i32 3
// COMMONIR:      [[TMP4:%.*]] = insertelement <8 x half> [[TMP3]], half %c, i32 4
// COMMONIR:      [[TMP5:%.*]] = insertelement <8 x half> [[TMP4]], half %c, i32 5
// COMMONIR:      [[TMP6:%.*]] = insertelement <8 x half> [[TMP5]], half %c, i32 6
// COMMONIR:      [[TMP7:%.*]] = insertelement <8 x half> [[TMP6]], half %c, i32 7
// UNCONSTRAINED: [[FMA:%.*]]  = call <8 x half> @llvm.fma.v8f16(<8 x half> [[SUB]], <8 x half> [[TMP7]], <8 x half> %a)
// CONSTRAINED:   [[FMA:%.*]]  = call <8 x half> @llvm.experimental.constrained.fma.v8f16(<8 x half> [[SUB]], <8 x half> [[TMP7]], <8 x half> %a, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// CHECK-ASM:     fmls v{{[0-9]+}}.8h, v{{[0-9]+}}.8h, v{{[0-9]+}}.h[{{[0-9]+}}]
// COMMONIR:      ret <8 x half> [[FMA]]
float16x8_t test_vfmsq_n_f16(float16x8_t a, float16x8_t b, float16_t c) {
  return vfmsq_n_f16(a, b, c);
}

// COMMON-LABEL: test_vfmsh_lane_f16
// UNCONSTRAINED: [[TMP0:%.*]] = fpext half %b to float
// CONSTRAINED:   [[TMP0:%.*]] = call float @llvm.experimental.constrained.fpext.f32.f16(half %b, metadata !"fpexcept.strict")
// CHECK-ASM:     fcvt s{{[0-9]+}}, h{{[0-9]+}}
// COMMONIR:      [[TMP1:%.*]] = fneg float [[TMP0]]
// CHECK-ASM:     fneg s{{[0-9]+}}, s{{[0-9]+}}
// UNCONSTRAINED: [[SUB:%.*]]  = fptrunc float [[TMP1]] to half
// CONSTRAINED:   [[SUB:%.*]]  = call half @llvm.experimental.constrained.fptrunc.f16.f32(float [[TMP1]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fcvt h{{[0-9]+}}, s{{[0-9]+}}
// COMMONIR:      [[EXTR:%.*]] = extractelement <4 x half> %c, i32 3
// UNCONSTRAINED: [[FMA:%.*]]  = call half @llvm.fma.f16(half [[SUB]], half [[EXTR]], half %a)
// CONSTRAINED:   [[FMA:%.*]]  = call half @llvm.experimental.constrained.fma.f16(half [[SUB]], half [[EXTR]], half %a, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// CHECK-ASM:     fmla h{{[0-9]+}}, h{{[0-9]+}}, v{{[0-9]+}}.h[{{[0-9]+}}]
// COMMONIR:      ret half [[FMA]]
float16_t test_vfmsh_lane_f16(float16_t a, float16_t b, float16x4_t c) {
  return vfmsh_lane_f16(a, b, c, 3);
}

// COMMON-LABEL: test_vfmsh_laneq_f16
// UNCONSTRAINED: [[TMP0:%.*]] = fpext half %b to float
// CONSTRAINED:   [[TMP0:%.*]] = call float @llvm.experimental.constrained.fpext.f32.f16(half %b, metadata !"fpexcept.strict")
// CHECK-ASM:     fcvt s{{[0-9]+}}, h{{[0-9]+}}
// COMMONIR:      [[TMP1:%.*]] = fneg float [[TMP0]]
// CHECK-ASM:     fneg s{{[0-9]+}}, s{{[0-9]+}}
// UNCONSTRAINED: [[SUB:%.*]]  = fptrunc float [[TMP1]] to half
// CONSTRAINED:   [[SUB:%.*]]  = call half @llvm.experimental.constrained.fptrunc.f16.f32(float [[TMP1]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fcvt h{{[0-9]+}}, s{{[0-9]+}}
// COMMONIR:      [[EXTR:%.*]] = extractelement <8 x half> %c, i32 7
// UNCONSTRAINED: [[FMA:%.*]]  = call half @llvm.fma.f16(half [[SUB]], half [[EXTR]], half %a)
// CONSTRAINED:   [[FMA:%.*]]  = call half @llvm.experimental.constrained.fma.f16(half [[SUB]], half [[EXTR]], half %a, metadata !"round.tonearest", metadata !"fpexcept.maytrap")
// CHECK-ASM:     fmla h{{[0-9]+}}, h{{[0-9]+}}, v{{[0-9]+}}.h[{{[0-9]+}}]
// COMMONIR:      ret half [[FMA]]
float16_t test_vfmsh_laneq_f16(float16_t a, float16_t b, float16x8_t c) {
  return vfmsh_laneq_f16(a, b, c, 7);
}
