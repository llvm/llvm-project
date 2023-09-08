// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-cpu cyclone \
// RUN: -disable-O0-optnone -emit-llvm -o - %s | opt -S -passes=mem2reg \
// RUN: | FileCheck --check-prefix=COMMON --check-prefix=COMMONIR --check-prefix=UNCONSTRAINED %s
// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-cpu cyclone \
// RUN: -ffp-exception-behavior=strict \
// RUN: -disable-O0-optnone -emit-llvm -o - %s | opt -S -passes=mem2reg \
// RUN: | FileCheck --check-prefix=COMMON --check-prefix=COMMONIR --check-prefix=CONSTRAINED %s
// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-cpu cyclone \
// RUN: -disable-O0-optnone -emit-llvm -o - %s | opt -S -passes=mem2reg | llc -o=- - \
// RUN: | FileCheck --check-prefix=COMMON --check-prefix=CHECK-ASM %s
// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon -target-cpu cyclone \
// RUN: -ffp-exception-behavior=strict \
// RUN: -disable-O0-optnone -emit-llvm -o - %s | opt -S -passes=mem2reg | llc -o=- - \
// RUN: | FileCheck --check-prefix=COMMON --check-prefix=CHECK-ASM %s

// REQUIRES: aarch64-registered-target

// Test new aarch64 intrinsics and types but constrained

#include <arm_neon.h>

// COMMON-LABEL: test_vfmas_lane_f32
// COMMONIR:        [[EXTRACT:%.*]] = extractelement <2 x float> %c, i32 1
// UNCONSTRAINED:   [[TMP2:%.*]] = call float @llvm.fma.f32(float %b, float [[EXTRACT]], float %a)
// CONSTRAINED:     [[TMP2:%.*]] = call float @llvm.experimental.constrained.fma.f32(float %b, float [[EXTRACT]], float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:       fmla s{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}.s[{{[0-9]+}}]
// COMMONIR:        ret float [[TMP2]]
float32_t test_vfmas_lane_f32(float32_t a, float32_t b, float32x2_t c) {
  return vfmas_lane_f32(a, b, c, 1);
}

// COMMON-LABEL: test_vfmad_lane_f64
// COMMONIR:        [[EXTRACT:%.*]] = extractelement <1 x double> %c, i32 0
// UNCONSTRAINED:   [[TMP2:%.*]] = call double @llvm.fma.f64(double %b, double [[EXTRACT]], double %a)
// CONSTRAINED:     [[TMP2:%.*]] = call double @llvm.experimental.constrained.fma.f64(double %b, double [[EXTRACT]], double %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:       fmadd d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
// COMMONIR:        ret double [[TMP2]]
float64_t test_vfmad_lane_f64(float64_t a, float64_t b, float64x1_t c) {
  return vfmad_lane_f64(a, b, c, 0);
}

// COMMON-LABEL: test_vfmad_laneq_f64
// COMMONIR:        [[EXTRACT:%.*]] = extractelement <2 x double> %c, i32 1
// UNCONSTRAINED:   [[TMP2:%.*]] = call double @llvm.fma.f64(double %b, double [[EXTRACT]], double %a)
// CONSTRAINED:     [[TMP2:%.*]] = call double @llvm.experimental.constrained.fma.f64(double %b, double [[EXTRACT]], double %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:       fmla d{{[0-9]+}}, d{{[0-9]+}}, v{{[0-9]+}}.d[{{[0-9]+}}]
// COMMONIR:        ret double [[TMP2]]
float64_t test_vfmad_laneq_f64(float64_t a, float64_t b, float64x2_t c) {
  return vfmad_laneq_f64(a, b, c, 1);
}

// COMMON-LABEL: test_vfmss_lane_f32
// COMMONIR:        [[SUB:%.*]] = fneg float %b
// COMMONIR:        [[EXTRACT:%.*]] = extractelement <2 x float> %c, i32 1
// UNCONSTRAINED:   [[TMP2:%.*]] = call float @llvm.fma.f32(float [[SUB]], float [[EXTRACT]], float %a)
// CONSTRAINED:     [[TMP2:%.*]] = call float @llvm.experimental.constrained.fma.f32(float [[SUB]], float [[EXTRACT]], float %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:       fmls s{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}.s[{{[0-9]+}}]
// COMMONIR:        ret float [[TMP2]]
float32_t test_vfmss_lane_f32(float32_t a, float32_t b, float32x2_t c) {
  return vfmss_lane_f32(a, b, c, 1);
}

// COMMON-LABEL: test_vfma_lane_f64
// COMMONIR:        [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// COMMONIR:        [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// COMMONIR:        [[TMP2:%.*]] = bitcast <1 x double> %v to <8 x i8>
// COMMONIR:        [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <1 x double>
// COMMONIR:        [[LANE:%.*]] = shufflevector <1 x double> [[TMP3]], <1 x double> [[TMP3]], <1 x i32> zeroinitializer
// COMMONIR:        [[FMLA:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x double>
// COMMONIR:        [[FMLA1:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x double>
// UNCONSTRAINED:   [[FMLA2:%.*]] = call <1 x double> @llvm.fma.v1f64(<1 x double> [[FMLA]], <1 x double> [[LANE]], <1 x double> [[FMLA1]])
// CONSTRAINED:     [[FMLA2:%.*]] = call <1 x double> @llvm.experimental.constrained.fma.v1f64(<1 x double> [[FMLA]], <1 x double> [[LANE]], <1 x double> [[FMLA1]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:       fmadd d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
// COMMONIR:        ret <1 x double> [[FMLA2]]
float64x1_t test_vfma_lane_f64(float64x1_t a, float64x1_t b, float64x1_t v) {
  return vfma_lane_f64(a, b, v, 0);
}

// COMMON-LABEL: test_vfms_lane_f64
// COMMONIR:        [[SUB:%.*]] = fneg <1 x double> %b
// COMMONIR:        [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// COMMONIR:        [[TMP1:%.*]] = bitcast <1 x double> [[SUB]] to <8 x i8>
// COMMONIR:        [[TMP2:%.*]] = bitcast <1 x double> %v to <8 x i8>
// COMMONIR:        [[TMP3:%.*]] = bitcast <8 x i8> [[TMP2]] to <1 x double>
// COMMONIR:        [[LANE:%.*]] = shufflevector <1 x double> [[TMP3]], <1 x double> [[TMP3]], <1 x i32> zeroinitializer
// COMMONIR:        [[FMLA:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x double>
// COMMONIR:        [[FMLA1:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x double>
// UNCONSTRAINED:   [[FMLA2:%.*]] = call <1 x double> @llvm.fma.v1f64(<1 x double> [[FMLA]], <1 x double> [[LANE]], <1 x double> [[FMLA1]])
// CONSTRAINED:     [[FMLA2:%.*]] = call <1 x double> @llvm.experimental.constrained.fma.v1f64(<1 x double> [[FMLA]], <1 x double> [[LANE]], <1 x double> [[FMLA1]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:       fmsub d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
// COMMONIR:        ret <1 x double> [[FMLA2]]
float64x1_t test_vfms_lane_f64(float64x1_t a, float64x1_t b, float64x1_t v) {
  return vfms_lane_f64(a, b, v, 0);
}

// COMMON-LABEL: test_vfma_laneq_f64
// COMMONIR:        [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// COMMONIR:        [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// COMMONIR:        [[TMP2:%.*]] = bitcast <2 x double> %v to <16 x i8>
// COMMONIR:        [[TMP3:%.*]] = bitcast <8 x i8> [[TMP0]] to double
// COMMONIR:        [[TMP4:%.*]] = bitcast <8 x i8> [[TMP1]] to double
// COMMONIR:        [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <2 x double>
// COMMONIR:        [[EXTRACT:%.*]] = extractelement <2 x double> [[TMP5]], i32 0
// UNCONSTRAINED:   [[TMP6:%.*]] = call double @llvm.fma.f64(double [[TMP4]], double [[EXTRACT]], double [[TMP3]])
// CONSTRAINED:     [[TMP6:%.*]] = call double @llvm.experimental.constrained.fma.f64(double [[TMP4]], double [[EXTRACT]], double [[TMP3]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:       fmadd d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
// COMMONIR:        [[TMP7:%.*]] = bitcast double [[TMP6]] to <1 x double>
// COMMONIR:        ret <1 x double> [[TMP7]]
float64x1_t test_vfma_laneq_f64(float64x1_t a, float64x1_t b, float64x2_t v) {
  return vfma_laneq_f64(a, b, v, 0);
}

// COMMON-LABEL: test_vfms_laneq_f64
// COMMONIR:        [[SUB:%.*]] = fneg <1 x double> %b
// CHECK-ASM:       fneg d{{[0-9]+}}, d{{[0-9]+}}
// COMMONIR:        [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// COMMONIR:        [[TMP1:%.*]] = bitcast <1 x double> [[SUB]] to <8 x i8>
// COMMONIR:        [[TMP2:%.*]] = bitcast <2 x double> %v to <16 x i8>
// COMMONIR:        [[TMP3:%.*]] = bitcast <8 x i8> [[TMP0]] to double
// COMMONIR:        [[TMP4:%.*]] = bitcast <8 x i8> [[TMP1]] to double
// COMMONIR:        [[TMP5:%.*]] = bitcast <16 x i8> [[TMP2]] to <2 x double>
// COMMONIR:        [[EXTRACT:%.*]] = extractelement <2 x double> [[TMP5]], i32 0
// UNCONSTRAINED:   [[TMP6:%.*]] = call double @llvm.fma.f64(double [[TMP4]], double [[EXTRACT]], double [[TMP3]])
// CONSTRAINED:     [[TMP6:%.*]] = call double @llvm.experimental.constrained.fma.f64(double [[TMP4]], double [[EXTRACT]], double [[TMP3]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:       fmadd d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
// COMMONIR:        [[TMP7:%.*]] = bitcast double [[TMP6]] to <1 x double>
// COMMONIR:        ret <1 x double> [[TMP7]]
float64x1_t test_vfms_laneq_f64(float64x1_t a, float64x1_t b, float64x2_t v) {
  return vfms_laneq_f64(a, b, v, 0);
}

