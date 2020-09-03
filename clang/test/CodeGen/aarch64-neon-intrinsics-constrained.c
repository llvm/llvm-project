// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:     -fallow-half-arguments-and-returns -S -disable-O0-optnone \
// RUN:  -flax-vector-conversions=none -emit-llvm -o - %s | opt -S -mem2reg \
// RUN: | FileCheck --check-prefix=COMMON --check-prefix=COMMONIR --check-prefix=UNCONSTRAINED %s
// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:     -fallow-half-arguments-and-returns -S -disable-O0-optnone \
// RUN:  -ffp-exception-behavior=strict \
// RUN:  -flax-vector-conversions=none -emit-llvm -o - %s | opt -S -mem2reg \
// RUN: | FileCheck --check-prefix=COMMON --check-prefix=COMMONIR --check-prefix=CONSTRAINED %s
// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:     -fallow-half-arguments-and-returns -S -disable-O0-optnone \
// RUN:  -flax-vector-conversions=none -o - %s \
// RUN: | FileCheck --check-prefix=COMMON --check-prefix=CHECK-ASM %s
// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:     -fallow-half-arguments-and-returns -S -disable-O0-optnone \
// RUN:  -ffp-exception-behavior=strict \
// RUN:  -flax-vector-conversions=none -o - %s \
// RUN: | FileCheck --check-prefix=COMMON --check-prefix=CHECK-ASM %s

// REQUIRES: aarch64-registered-target

// Fails during instruction selection:
// XFAIL: *

// Test new aarch64 intrinsics and types but constrained

#include <arm_neon.h>

// COMMON-LABEL: test_vadd_f32
// UNCONSTRAINED: [[ADD_I:%.*]] = fadd <2 x float> %v1, %v2
// CONSTRAINED:   [[ADD_I:%.*]] = call <2 x float> @llvm.experimental.constrained.fadd.v2f32(<2 x float> %v1, <2 x float> %v2, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fadd v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
// COMMONIR:      ret <2 x float> [[ADD_I]]
float32x2_t test_vadd_f32(float32x2_t v1, float32x2_t v2) {
  return vadd_f32(v1, v2);
}

// COMMON-LABEL: test_vaddq_f32
// UNCONSTRAINED: [[ADD_I:%.*]] = fadd <4 x float> %v1, %v2
// CONSTRAINED:   [[ADD_I:%.*]] = call <4 x float> @llvm.experimental.constrained.fadd.v4f32(<4 x float> %v1, <4 x float> %v2, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fadd v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
// COMMONIR:      ret <4 x float> [[ADD_I]]
float32x4_t test_vaddq_f32(float32x4_t v1, float32x4_t v2) {
  return vaddq_f32(v1, v2);
}

// COMMON-LABEL: test_vsub_f32
// UNCONSTRAINED: [[SUB_I:%.*]] = fsub <2 x float> %v1, %v2
// CONSTRAINED:   [[SUB_I:%.*]] = call <2 x float> @llvm.experimental.constrained.fsub.v2f32(<2 x float> %v1, <2 x float> %v2, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fsub v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
// COMMONIR:      ret <2 x float> [[SUB_I]]
float32x2_t test_vsub_f32(float32x2_t v1, float32x2_t v2) {
  return vsub_f32(v1, v2);
}

// COMMON-LABEL: test_vsubq_f32
// UNCONSTRAINED: [[SUB_I:%.*]] = fsub <4 x float> %v1, %v2
// CONSTRAINED:   [[SUB_I:%.*]] = call <4 x float> @llvm.experimental.constrained.fsub.v4f32(<4 x float> %v1, <4 x float> %v2, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fsub v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
// COMMONIR:      ret <4 x float> [[SUB_I]]
float32x4_t test_vsubq_f32(float32x4_t v1, float32x4_t v2) {
  return vsubq_f32(v1, v2);
}

// COMMON-LABEL: test_vsubq_f64
// UNCONSTRAINED: [[SUB_I:%.*]] = fsub <2 x double> %v1, %v2
// CONSTRAINED:   [[SUB_I:%.*]] = call <2 x double> @llvm.experimental.constrained.fsub.v2f64(<2 x double> %v1, <2 x double> %v2, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fsub v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
// COMMONIR:      ret <2 x double> [[SUB_I]]
float64x2_t test_vsubq_f64(float64x2_t v1, float64x2_t v2) {
  return vsubq_f64(v1, v2);
}

// COMMON-LABEL: test_vmul_f32
// UNCONSTRAINED: [[MUL_I:%.*]] = fmul <2 x float> %v1, %v2
// CONSTRAINED:   [[MUL_I:%.*]] = call <2 x float> @llvm.experimental.constrained.fmul.v2f32(<2 x float> %v1, <2 x float> %v2, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fmul v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
// COMMONIR:      ret <2 x float> [[MUL_I]]
float32x2_t test_vmul_f32(float32x2_t v1, float32x2_t v2) {
  return vmul_f32(v1, v2);
}

// COMMON-LABEL: test_vmulq_f32
// UNCONSTRAINED: [[MUL_I:%.*]] = fmul <4 x float> %v1, %v2
// CONSTRAINED:   [[MUL_I:%.*]] = call <4 x float> @llvm.experimental.constrained.fmul.v4f32(<4 x float> %v1, <4 x float> %v2, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fmul v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
// COMMONIR:      ret <4 x float> [[MUL_I]]
float32x4_t test_vmulq_f32(float32x4_t v1, float32x4_t v2) {
  return vmulq_f32(v1, v2);
}

// COMMON-LABEL: test_vmulq_f64
// UNCONSTRAINED: [[MUL_I:%.*]] = fmul <2 x double> %v1, %v2
// CONSTRAINED:   [[MUL_I:%.*]] = call <2 x double> @llvm.experimental.constrained.fmul.v2f64(<2 x double> %v1, <2 x double> %v2, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fmul v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
// COMMONIR:      ret <2 x double> [[MUL_I]]
float64x2_t test_vmulq_f64(float64x2_t v1, float64x2_t v2) {
  return vmulq_f64(v1, v2);
}

// COMMON-LABEL: test_vmla_f32
// UNCONSTRAINED: [[MUL_I:%.*]] = fmul <2 x float> %v2, %v3
// CONSTRAINED:   [[MUL_I:%.*]] = call <2 x float> @llvm.experimental.constrained.fmul.v2f32(<2 x float> %v2, <2 x float> %v3, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fmul [[MUL_R:v[0-9]+.2s]], v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
// UNCONSTRAINED: [[ADD_I:%.*]] = fadd <2 x float> %v1, [[MUL_I]]
// CONSTRAINED:   [[ADD_I:%.*]] = call <2 x float> @llvm.experimental.constrained.fadd.v2f32(<2 x float> %v1, <2 x float> [[MUL_I]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM-NEXT:fadd v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, [[MUL_R]]
// COMMONIR:      ret <2 x float> [[ADD_I]]
float32x2_t test_vmla_f32(float32x2_t v1, float32x2_t v2, float32x2_t v3) {
  return vmla_f32(v1, v2, v3);
}

// COMMON-LABEL: test_vmlaq_f32
// UNCONSTRAINED: [[MUL_I:%.*]] = fmul <4 x float> %v2, %v3
// CONSTRAINED:   [[MUL_I:%.*]] = call <4 x float> @llvm.experimental.constrained.fmul.v4f32(<4 x float> %v2, <4 x float> %v3, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fmul [[MUL_R:v[0-9]+.4s]], v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
// UNCONSTRAINED: [[ADD_I:%.*]] = fadd <4 x float> %v1, [[MUL_I]]
// CONSTRAINED:   [[ADD_I:%.*]] = call <4 x float> @llvm.experimental.constrained.fadd.v4f32(<4 x float> %v1, <4 x float> [[MUL_I]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM-NEXT:fadd v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, [[MUL_R]]
// COMMONIR:      ret <4 x float> [[ADD_I]]
float32x4_t test_vmlaq_f32(float32x4_t v1, float32x4_t v2, float32x4_t v3) {
  return vmlaq_f32(v1, v2, v3);
}

// COMMON-LABEL: test_vmlaq_f64
// UNCONSTRAINED: [[MUL_I:%.*]] = fmul <2 x double> %v2, %v3
// CONSTRAINED:   [[MUL_I:%.*]] = call <2 x double> @llvm.experimental.constrained.fmul.v2f64(<2 x double> %v2, <2 x double> %v3, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fmul [[MUL_R:v[0-9]+.2d]], v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
// UNCONSTRAINED: [[ADD_I:%.*]] = fadd <2 x double> %v1, [[MUL_I]]
// CONSTRAINED:   [[ADD_I:%.*]] = call <2 x double> @llvm.experimental.constrained.fadd.v2f64(<2 x double> %v1, <2 x double> [[MUL_I]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM-NEXT:fadd v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, [[MUL_R]]
// COMMONIR:      ret <2 x double> [[ADD_I]]
float64x2_t test_vmlaq_f64(float64x2_t v1, float64x2_t v2, float64x2_t v3) {
  return vmlaq_f64(v1, v2, v3);
}

// COMMON-LABEL: test_vmls_f32
// UNCONSTRAINED: [[MUL_I:%.*]] = fmul <2 x float> %v2, %v3
// CONSTRAINED:   [[MUL_I:%.*]] = call <2 x float> @llvm.experimental.constrained.fmul.v2f32(<2 x float> %v2, <2 x float> %v3, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fmul [[MUL_R:v[0-9]+.2s]], v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
// UNCONSTRAINED: [[SUB_I:%.*]] = fsub <2 x float> %v1, [[MUL_I]]
// CONSTRAINED:   [[SUB_I:%.*]] = call <2 x float> @llvm.experimental.constrained.fsub.v2f32(<2 x float> %v1, <2 x float> [[MUL_I]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM-NEXT:fsub v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, [[MUL_R]]
// COMMONIR:      ret <2 x float> [[SUB_I]]
float32x2_t test_vmls_f32(float32x2_t v1, float32x2_t v2, float32x2_t v3) {
  return vmls_f32(v1, v2, v3);
}

// COMMON-LABEL: test_vmlsq_f32
// UNCONSTRAINED: [[MUL_I:%.*]] = fmul <4 x float> %v2, %v3
// CONSTRAINED:   [[MUL_I:%.*]] = call <4 x float> @llvm.experimental.constrained.fmul.v4f32(<4 x float> %v2, <4 x float> %v3, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fmul [[MUL_R:v[0-9]+.4s]], v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
// UNCONSTRAINED: [[SUB_I:%.*]] = fsub <4 x float> %v1, [[MUL_I]]
// CONSTRAINED:   [[SUB_I:%.*]] = call <4 x float> @llvm.experimental.constrained.fsub.v4f32(<4 x float> %v1, <4 x float> [[MUL_I]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM-NEXT:fsub v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, [[MUL_R]]
// COMMONIR:   ret <4 x float> [[SUB_I]]
float32x4_t test_vmlsq_f32(float32x4_t v1, float32x4_t v2, float32x4_t v3) {
  return vmlsq_f32(v1, v2, v3);
}

// COMMON-LABEL: test_vmlsq_f64
// UNCONSTRAINED: [[MUL_I:%.*]] = fmul <2 x double> %v2, %v3
// CONSTRAINED:   [[MUL_I:%.*]] = call <2 x double> @llvm.experimental.constrained.fmul.v2f64(<2 x double> %v2, <2 x double> %v3, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fmul [[MUL_R:v[0-9]+.2d]], v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
// UNCONSTRAINED: [[SUB_I:%.*]] = fsub <2 x double> %v1, [[MUL_I]]
// CONSTRAINED:   [[SUB_I:%.*]] = call <2 x double> @llvm.experimental.constrained.fsub.v2f64(<2 x double> %v1, <2 x double> [[MUL_I]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM-NEXT:fsub v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, [[MUL_R]]
// COMMONIR:      ret <2 x double> [[SUB_I]]
float64x2_t test_vmlsq_f64(float64x2_t v1, float64x2_t v2, float64x2_t v3) {
  return vmlsq_f64(v1, v2, v3);
}

// COMMON-LABEL: test_vfma_f32
// COMMONIR:      [[TMP0:%.*]] = bitcast <2 x float> %v1 to <8 x i8>
// COMMONIR:      [[TMP1:%.*]] = bitcast <2 x float> %v2 to <8 x i8>
// COMMONIR:      [[TMP2:%.*]] = bitcast <2 x float> %v3 to <8 x i8>
// UNCONSTRAINED: [[TMP3:%.*]] = call <2 x float> @llvm.fma.v2f32(<2 x float> %v2, <2 x float> %v3, <2 x float> %v1)
// CONSTRAINED:   [[TMP3:%.*]] = call <2 x float> @llvm.experimental.constrained.fma.v2f32(<2 x float> %v2, <2 x float> %v3, <2 x float> %v1, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fmla v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
// COMMONIR:      ret <2 x float> [[TMP3]]
float32x2_t test_vfma_f32(float32x2_t v1, float32x2_t v2, float32x2_t v3) {
  return vfma_f32(v1, v2, v3);
}

// COMMON-LABEL: test_vfmaq_f32
// COMMONIR:      [[TMP0:%.*]] = bitcast <4 x float> %v1 to <16 x i8>
// COMMONIR:      [[TMP1:%.*]] = bitcast <4 x float> %v2 to <16 x i8>
// COMMONIR:      [[TMP2:%.*]] = bitcast <4 x float> %v3 to <16 x i8>
// UNCONSTRAINED: [[TMP3:%.*]] = call <4 x float> @llvm.fma.v4f32(<4 x float> %v2, <4 x float> %v3, <4 x float> %v1)
// CONSTRAINED:   [[TMP3:%.*]] = call <4 x float> @llvm.experimental.constrained.fma.v4f32(<4 x float> %v2, <4 x float> %v3, <4 x float> %v1, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fmla v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
// COMMONIR:      ret <4 x float> [[TMP3]]
float32x4_t test_vfmaq_f32(float32x4_t v1, float32x4_t v2, float32x4_t v3) {
  return vfmaq_f32(v1, v2, v3);
}

// COMMON-LABEL: test_vfmaq_f64
// COMMONIR:      [[TMP0:%.*]] = bitcast <2 x double> %v1 to <16 x i8>
// COMMONIR:      [[TMP1:%.*]] = bitcast <2 x double> %v2 to <16 x i8>
// COMMONIR:      [[TMP2:%.*]] = bitcast <2 x double> %v3 to <16 x i8>
// UNCONSTRAINED: [[TMP3:%.*]] = call <2 x double> @llvm.fma.v2f64(<2 x double> %v2, <2 x double> %v3, <2 x double> %v1)
// CONSTRAINED:   [[TMP3:%.*]] = call <2 x double> @llvm.experimental.constrained.fma.v2f64(<2 x double> %v2, <2 x double> %v3, <2 x double> %v1, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fmla v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
// COMMONIR:      ret <2 x double> [[TMP3]]
float64x2_t test_vfmaq_f64(float64x2_t v1, float64x2_t v2, float64x2_t v3) {
  return vfmaq_f64(v1, v2, v3);
}

// COMMON-LABEL: test_vfms_f32
// COMMONIR:      [[SUB_I:%.*]] = fneg <2 x float> %v2
// CHECK-ASM:     fneg v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
// COMMONIR:      [[TMP0:%.*]] = bitcast <2 x float> %v1 to <8 x i8>
// COMMONIR:      [[TMP1:%.*]] = bitcast <2 x float> [[SUB_I]] to <8 x i8>
// COMMONIR:      [[TMP2:%.*]] = bitcast <2 x float> %v3 to <8 x i8>
// UNCONSTRAINED: [[TMP3:%.*]] = call <2 x float> @llvm.fma.v2f32(<2 x float> [[SUB_I]], <2 x float> %v3, <2 x float> %v1)
// CONSTRAINED:   [[TMP3:%.*]] = call <2 x float> @llvm.experimental.constrained.fma.v2f32(<2 x float> [[SUB_I]], <2 x float> %v3, <2 x float> %v1, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fmla v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
// COMMONIR:      ret <2 x float> [[TMP3]]
float32x2_t test_vfms_f32(float32x2_t v1, float32x2_t v2, float32x2_t v3) {
  return vfms_f32(v1, v2, v3);
}

// COMMON-LABEL: test_vfmsq_f32
// COMMONIR:      [[SUB_I:%.*]] = fneg <4 x float> %v2
// CHECK-ASM:     fneg v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
// COMMONIR:      [[TMP0:%.*]] = bitcast <4 x float> %v1 to <16 x i8>
// COMMONIR:      [[TMP1:%.*]] = bitcast <4 x float> [[SUB_I]] to <16 x i8>
// COMMONIR:      [[TMP2:%.*]] = bitcast <4 x float> %v3 to <16 x i8>
// UNCONSTRAINED: [[TMP3:%.*]] = call <4 x float> @llvm.fma.v4f32(<4 x float> [[SUB_I]], <4 x float> %v3, <4 x float> %v1)
// CONSTRAINED:   [[TMP3:%.*]] = call <4 x float> @llvm.experimental.constrained.fma.v4f32(<4 x float> [[SUB_I]], <4 x float> %v3, <4 x float> %v1, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fmla v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
// COMMONIR:      ret <4 x float> [[TMP3]]
float32x4_t test_vfmsq_f32(float32x4_t v1, float32x4_t v2, float32x4_t v3) {
  return vfmsq_f32(v1, v2, v3);
}

// COMMON-LABEL: test_vfmsq_f64
// COMMONIR:      [[SUB_I:%.*]] = fneg <2 x double> %v2
// CHECK-ASM:     fneg v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
// COMMONIR:      [[TMP0:%.*]] = bitcast <2 x double> %v1 to <16 x i8>
// COMMONIR:      [[TMP1:%.*]] = bitcast <2 x double> [[SUB_I]] to <16 x i8>
// COMMONIR:      [[TMP2:%.*]] = bitcast <2 x double> %v3 to <16 x i8>
// UNCONSTRAINED: [[TMP3:%.*]] = call <2 x double> @llvm.fma.v2f64(<2 x double> [[SUB_I]], <2 x double> %v3, <2 x double> %v1)
// CONSTRAINED:   [[TMP3:%.*]] = call <2 x double> @llvm.experimental.constrained.fma.v2f64(<2 x double> [[SUB_I]], <2 x double> %v3, <2 x double> %v1, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fmla v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
// COMMONIR:      ret <2 x double> [[TMP3]]
float64x2_t test_vfmsq_f64(float64x2_t v1, float64x2_t v2, float64x2_t v3) {
  return vfmsq_f64(v1, v2, v3);
}

// COMMON-LABEL: test_vdivq_f64
// UNCONSTRAINED: [[DIV_I:%.*]] = fdiv <2 x double> %v1, %v2
// CONSTRAINED:   [[DIV_I:%.*]] = call <2 x double> @llvm.experimental.constrained.fdiv.v2f64(<2 x double> %v1, <2 x double> %v2, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fdiv v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
// COMMONIR:      ret <2 x double> [[DIV_I]]
float64x2_t test_vdivq_f64(float64x2_t v1, float64x2_t v2) {
  return vdivq_f64(v1, v2);
}

// COMMON-LABEL: test_vdivq_f32
// UNCONSTRAINED: [[DIV_I:%.*]] = fdiv <4 x float> %v1, %v2
// CONSTRAINED:   [[DIV_I:%.*]] = call <4 x float> @llvm.experimental.constrained.fdiv.v4f32(<4 x float> %v1, <4 x float> %v2, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fdiv v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
// COMMONIR:      ret <4 x float> [[DIV_I]]
float32x4_t test_vdivq_f32(float32x4_t v1, float32x4_t v2) {
  return vdivq_f32(v1, v2);
}

// COMMON-LABEL: test_vdiv_f32
// UNCONSTRAINED: [[DIV_I:%.*]] = fdiv <2 x float> %v1, %v2
// CONSTRAINED:   [[DIV_I:%.*]] = call <2 x float> @llvm.experimental.constrained.fdiv.v2f32(<2 x float> %v1, <2 x float> %v2, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fdiv v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
// COMMONIR:      ret <2 x float> [[DIV_I]]
float32x2_t test_vdiv_f32(float32x2_t v1, float32x2_t v2) {
  return vdiv_f32(v1, v2);
}

// COMMON-LABEL: test_vceq_f32
// UNCONSTRAINED: [[CMP_I:%.*]] = fcmp oeq <2 x float> %v1, %v2
// CONSTRAINED:   [[CMP_I:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmp.v2f32(<2 x float> %v1, <2 x float> %v2, metadata !"oeq", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmeq v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
// COMMONIR:      [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// COMMONIR:      ret <2 x i32> [[SEXT_I]]
uint32x2_t test_vceq_f32(float32x2_t v1, float32x2_t v2) {
  return vceq_f32(v1, v2);
}

// COMMON-LABEL: test_vceq_f64
// UNCONSTRAINED: [[CMP_I:%.*]] = fcmp oeq <1 x double> %a, %b
// CONSTRAINED:   [[CMP_I:%.*]] = call <1 x i1> @llvm.experimental.constrained.fcmp.v1f64(<1 x double> %a, <1 x double> %b, metadata !"oeq", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmp d{{[0-9]+}}, d{{[0-9]+}}
// CHECK-ASM-NEXT:cset {{w[0-9]+}}, eq
// COMMONIR:      [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// COMMONIR:      ret <1 x i64> [[SEXT_I]]
uint64x1_t test_vceq_f64(float64x1_t a, float64x1_t b) {
  return vceq_f64(a, b);
}

// COMMON-LABEL: test_vceqq_f32
// UNCONSTRAINED: [[CMP_I:%.*]] = fcmp oeq <4 x float> %v1, %v2
// CONSTRAINED:   [[CMP_I:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %v1, <4 x float> %v2, metadata !"oeq", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmeq v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
// COMMONIR:      [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// COMMONIR:      ret <4 x i32> [[SEXT_I]]
uint32x4_t test_vceqq_f32(float32x4_t v1, float32x4_t v2) {
  return vceqq_f32(v1, v2);
}

// COMMON-LABEL: test_vceqq_f64
// UNCONSTRAINED: [[CMP_I:%.*]] = fcmp oeq <2 x double> %v1, %v2
// CONSTRAINED:   [[CMP_I:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %v1, <2 x double> %v2, metadata !"oeq", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmeq v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
// COMMONIR:      [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// COMMONIR:      ret <2 x i64> [[SEXT_I]]
uint64x2_t test_vceqq_f64(float64x2_t v1, float64x2_t v2) {
  return vceqq_f64(v1, v2);
}

// COMMON-LABEL: test_vcge_f32
// UNCONSTRAINED: [[CMP_I:%.*]] = fcmp oge <2 x float> %v1, %v2
// CONSTRAINED:   [[CMP_I:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f32(<2 x float> %v1, <2 x float> %v2, metadata !"oge", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmge v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
// COMMONIR:      [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// COMMONIR:      ret <2 x i32> [[SEXT_I]]
uint32x2_t test_vcge_f32(float32x2_t v1, float32x2_t v2) {
  return vcge_f32(v1, v2);
}

// COMMON-LABEL: test_vcge_f64
// UNCONSTRAINED: [[CMP_I:%.*]] = fcmp oge <1 x double> %a, %b
// CONSTRAINED:   [[CMP_I:%.*]] = call <1 x i1> @llvm.experimental.constrained.fcmps.v1f64(<1 x double> %a, <1 x double> %b, metadata !"oge", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmp d{{[0-9]+}}, d{{[0-9]+}}
// CHECK-ASM-NEXT:cset {{w[0-9]+}}, ge
// COMMONIR:      [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// COMMONIR:      ret <1 x i64> [[SEXT_I]]
uint64x1_t test_vcge_f64(float64x1_t a, float64x1_t b) {
  return vcge_f64(a, b);
}

// COMMON-LABEL: test_vcgeq_f32
// UNCONSTRAINED: [[CMP_I:%.*]] = fcmp oge <4 x float> %v1, %v2
// CONSTRAINED:   [[CMP_I:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %v1, <4 x float> %v2, metadata !"oge", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmge v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
// COMMONIR:      [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// COMMONIR:      ret <4 x i32> [[SEXT_I]]
uint32x4_t test_vcgeq_f32(float32x4_t v1, float32x4_t v2) {
  return vcgeq_f32(v1, v2);
}

// COMMON-LABEL: test_vcgeq_f64
// UNCONSTRAINED: [[CMP_I:%.*]] = fcmp oge <2 x double> %v1, %v2
// CONSTRAINED:   [[CMP_I:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %v1, <2 x double> %v2, metadata !"oge", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmge v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
// COMMONIR:      [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// COMMONIR:      ret <2 x i64> [[SEXT_I]]
uint64x2_t test_vcgeq_f64(float64x2_t v1, float64x2_t v2) {
  return vcgeq_f64(v1, v2);
}

// COMMON-LABEL: test_vcle_f32
// UNCONSTRAINED: [[CMP_I:%.*]] = fcmp ole <2 x float> %v1, %v2
// CONSTRAINED:   [[CMP_I:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f32(<2 x float> %v1, <2 x float> %v2, metadata !"ole", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmge v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
// COMMONIR:      [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// COMMONIR:      ret <2 x i32> [[SEXT_I]]
uint32x2_t test_vcle_f32(float32x2_t v1, float32x2_t v2) {
  return vcle_f32(v1, v2);
}

// COMMON-LABEL: test_vcle_f64
// UNCONSTRAINED: [[CMP_I:%.*]] = fcmp ole <1 x double> %a, %b
// CONSTRAINED:   [[CMP_I:%.*]] = call <1 x i1> @llvm.experimental.constrained.fcmps.v1f64(<1 x double> %a, <1 x double> %b, metadata !"ole", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmp d{{[0-9]+}}, d{{[0-9]+}}
// CHECK-ASM-NEXT:cset {{w[0-9]+}}, ls
// COMMONIR:      [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// COMMONIR:      ret <1 x i64> [[SEXT_I]]
uint64x1_t test_vcle_f64(float64x1_t a, float64x1_t b) {
  return vcle_f64(a, b);
}

// COMMON-LABEL: test_vcleq_f32
// UNCONSTRAINED: [[CMP_I:%.*]] = fcmp ole <4 x float> %v1, %v2
// CONSTRAINED:   [[CMP_I:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %v1, <4 x float> %v2, metadata !"ole", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmge v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
// COMMONIR:      [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// COMMONIR:      ret <4 x i32> [[SEXT_I]]
uint32x4_t test_vcleq_f32(float32x4_t v1, float32x4_t v2) {
  return vcleq_f32(v1, v2);
}

// COMMON-LABEL: test_vcleq_f64
// UNCONSTRAINED: [[CMP_I:%.*]] = fcmp ole <2 x double> %v1, %v2
// CONSTRAINED:   [[CMP_I:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %v1, <2 x double> %v2, metadata !"ole", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmge v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
// COMMONIR:      [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// COMMONIR:      ret <2 x i64> [[SEXT_I]]
uint64x2_t test_vcleq_f64(float64x2_t v1, float64x2_t v2) {
  return vcleq_f64(v1, v2);
}

// COMMON-LABEL: test_vcgt_f32
// UNCONSTRAINED: [[CMP_I:%.*]] = fcmp ogt <2 x float> %v1, %v2
// CONSTRAINED:   [[CMP_I:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f32(<2 x float> %v1, <2 x float> %v2, metadata !"ogt", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmgt v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
// COMMONIR:      [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// COMMONIR:      ret <2 x i32> [[SEXT_I]]
uint32x2_t test_vcgt_f32(float32x2_t v1, float32x2_t v2) {
  return vcgt_f32(v1, v2);
}

// COMMON-LABEL: test_vcgt_f64
// UNCONSTRAINED: [[CMP_I:%.*]] = fcmp ogt <1 x double> %a, %b
// CONSTRAINED:   [[CMP_I:%.*]] = call <1 x i1> @llvm.experimental.constrained.fcmps.v1f64(<1 x double> %a, <1 x double> %b, metadata !"ogt", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmp d{{[0-9]+}}, d{{[0-9]+}}
// CHECK-ASM-NEXT:cset {{w[0-9]+}}, gt
// COMMONIR:      [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// COMMONIR:      ret <1 x i64> [[SEXT_I]]
uint64x1_t test_vcgt_f64(float64x1_t a, float64x1_t b) {
  return vcgt_f64(a, b);
}

// COMMON-LABEL: test_vcgtq_f32
// UNCONSTRAINED: [[CMP_I:%.*]] = fcmp ogt <4 x float> %v1, %v2
// CONSTRAINED:   [[CMP_I:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %v1, <4 x float> %v2, metadata !"ogt", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmgt v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
// COMMONIR:      [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// COMMONIR:      ret <4 x i32> [[SEXT_I]]
uint32x4_t test_vcgtq_f32(float32x4_t v1, float32x4_t v2) {
  return vcgtq_f32(v1, v2);
}

// COMMON-LABEL: test_vcgtq_f64
// UNCONSTRAINED: [[CMP_I:%.*]] = fcmp ogt <2 x double> %v1, %v2
// CONSTRAINED:   [[CMP_I:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %v1, <2 x double> %v2, metadata !"ogt", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmgt v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
// COMMONIR:      [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// COMMONIR:      ret <2 x i64> [[SEXT_I]]
uint64x2_t test_vcgtq_f64(float64x2_t v1, float64x2_t v2) {
  return vcgtq_f64(v1, v2);
}

// COMMON-LABEL: test_vclt_f32
// UNCONSTRAINED: [[CMP_I:%.*]] = fcmp olt <2 x float> %v1, %v2
// CONSTRAINED:   [[CMP_I:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f32(<2 x float> %v1, <2 x float> %v2, metadata !"olt", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmgt v{{[0-9]+}}.2s, v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
// COMMONIR:      [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// COMMONIR:      ret <2 x i32> [[SEXT_I]]
uint32x2_t test_vclt_f32(float32x2_t v1, float32x2_t v2) {
  return vclt_f32(v1, v2);
}

// COMMON-LABEL: test_vclt_f64
// UNCONSTRAINED: [[CMP_I:%.*]] = fcmp olt <1 x double> %a, %b
// CONSTRAINED:   [[CMP_I:%.*]] = call <1 x i1> @llvm.experimental.constrained.fcmps.v1f64(<1 x double> %a, <1 x double> %b, metadata !"olt", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmp d{{[0-9]+}}, d{{[0-9]+}}
// CHECK-ASM-NEXT:cset {{w[0-9]+}}, mi
// COMMONIR:      [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// COMMONIR:      ret <1 x i64> [[SEXT_I]]
uint64x1_t test_vclt_f64(float64x1_t a, float64x1_t b) {
  return vclt_f64(a, b);
}

// COMMON-LABEL: test_vcltq_f32
// UNCONSTRAINED: [[CMP_I:%.*]] = fcmp olt <4 x float> %v1, %v2
// CONSTRAINED:   [[CMP_I:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %v1, <4 x float> %v2, metadata !"olt", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmgt v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
// COMMONIR:      [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// COMMONIR:      ret <4 x i32> [[SEXT_I]]
uint32x4_t test_vcltq_f32(float32x4_t v1, float32x4_t v2) {
  return vcltq_f32(v1, v2);
}

// COMMON-LABEL: test_vcltq_f64
// UNCONSTRAINED: [[CMP_I:%.*]] = fcmp olt <2 x double> %v1, %v2
// CONSTRAINED:   [[CMP_I:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %v1, <2 x double> %v2, metadata !"olt", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmgt v{{[0-9]+}}.2d, v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
// COMMONIR:      [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// COMMONIR:      ret <2 x i64> [[SEXT_I]]
uint64x2_t test_vcltq_f64(float64x2_t v1, float64x2_t v2) {
  return vcltq_f64(v1, v2);
}

// COMMON-LABEL: test_vpadds_f32
// COMMONIR:      [[LANE0_I:%.*]] = extractelement <2 x float> %a, i64 0
// COMMONIR:      [[LANE1_I:%.*]] = extractelement <2 x float> %a, i64 1
// UNCONSTRAINED: [[VPADDD_I:%.*]] = fadd float [[LANE0_I]], [[LANE1_I]]
// CONSTRAINED:   [[VPADDD_I:%.*]] = call float @llvm.experimental.constrained.fadd.f32(float [[LANE0_I]], float [[LANE1_I]], metadata !"round.tonearest", metadata !"fpexcept.strict"
// CHECK-ASM:     fadd s{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
// COMMONIR:      ret float [[VPADDD_I]]
float32_t test_vpadds_f32(float32x2_t a) {
  return vpadds_f32(a);
}

// COMMON-LABEL: test_vpaddd_f64
// COMMONIR:      [[LANE0_I:%.*]] = extractelement <2 x double> %a, i64 0
// COMMONIR:      [[LANE1_I:%.*]] = extractelement <2 x double> %a, i64 1
// UNCONSTRAINED: [[VPADDD_I:%.*]] = fadd double [[LANE0_I]], [[LANE1_I]]
// CONSTRAINED:   [[VPADDD_I:%.*]] = call double @llvm.experimental.constrained.fadd.f64(double [[LANE0_I]], double [[LANE1_I]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     faddp d{{[0-9]+}}, v{{[0-9]+}}.2d
// COMMONIR:      ret double [[VPADDD_I]]
float64_t test_vpaddd_f64(float64x2_t a) {
  return vpaddd_f64(a);
}

// COMMON-LABEL: test_vcvts_f32_s32
// UNCONSTRAINED: [[TMP0:%.*]] = sitofp i32 %a to float
// CONSTRAINED:   [[TMP0:%.*]] = call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     scvtf s{{[0-9]+}}, w{{[0-9]+}}
// COMMONIR:      ret float [[TMP0]]
float32_t test_vcvts_f32_s32(int32_t a) {
  return vcvts_f32_s32(a);
}

// COMMON-LABEL: test_vcvtd_f64_s64
// UNCONSTRAINED: [[TMP0:%.*]] = sitofp i64 %a to double
// CONSTRAINED:   [[TMP0:%.*]] = call double @llvm.experimental.constrained.sitofp.f64.i64(i64 %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     scvtf d{{[0-9]}}, x{{[0-9]+}}
// COMMONIR:      ret double [[TMP0]]
float64_t test_vcvtd_f64_s64(int64_t a) {
  return vcvtd_f64_s64(a);
}

// COMMON-LABEL: test_vcvts_f32_u32
// UNCONSTRAINED: [[TMP0:%.*]] = uitofp i32 %a to float
// CONSTRAINED:   [[TMP0:%.*]] = call float @llvm.experimental.constrained.uitofp.f32.i32(i32 %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     ucvtf s{{[0-9]+}}, w{{[0-9]+}}
// COMMONIR:      ret float [[TMP0]]
float32_t test_vcvts_f32_u32(uint32_t a) {
  return vcvts_f32_u32(a);
}

// XXX should verify the type of registers
// COMMON-LABEL: test_vcvtd_f64_u64
// UNCONSTRAINED: [[TMP0:%.*]] = uitofp i64 %a to double
// CONSTRAINED:   [[TMP0:%.*]] = call double @llvm.experimental.constrained.uitofp.f64.i64(i64 %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     ucvtf d{{[0-9]}}, x{{[0-9]+}}
// COMMONIR:      ret double [[TMP0]]
float64_t test_vcvtd_f64_u64(uint64_t a) {
  return vcvtd_f64_u64(a);
}

// COMMON-LABEL: test_vceqs_f32
// UNCONSTRAINED: [[TMP0:%.*]] = fcmp oeq float %a, %b
// CONSTRAINED:   [[TMP0:%.*]] = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"oeq", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmp s{{[0-9]+}}, s{{[0-9]+}}
// CHECK-ASM-NEXT:cset {{w[0-9]+}}, eq
// COMMONIR:      [[VCMPD_I:%.*]] = sext i1 [[TMP0]] to i32
// COMMONIR:      ret i32 [[VCMPD_I]]
uint32_t test_vceqs_f32(float32_t a, float32_t b) {
  return (uint32_t)vceqs_f32(a, b);
}

// COMMON-LABEL: test_vceqd_f64
// UNCONSTRAINED: [[TMP0:%.*]] = fcmp oeq double %a, %b
// CONSTRAINED:   [[TMP0:%.*]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"oeq", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmp d{{[0-9]+}}, d{{[0-9]+}}
// CHECK-ASM-NEXT:cset {{w[0-9]+}}, eq
// COMMONIR:      [[VCMPD_I:%.*]] = sext i1 [[TMP0]] to i64
// COMMONIR:      ret i64 [[VCMPD_I]]
uint64_t test_vceqd_f64(float64_t a, float64_t b) {
  return (uint64_t)vceqd_f64(a, b);
}

// COMMON-LABEL: test_vceqzs_f32
// UNCONSTRAINED: [[TMP0:%.*]] = fcmp oeq float %a, 0.000000e+00
// CONSTRAINED:   [[TMP0:%.*]] = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float 0.000000e+00, metadata !"oeq", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmp s{{[0-9]+}}, #0.0
// CHECK-ASM-NEXT:cset {{w[0-9]+}}, eq
// COMMONIR:      [[VCEQZ_I:%.*]] = sext i1 [[TMP0]] to i32
// COMMONIR:      ret i32 [[VCEQZ_I]]
uint32_t test_vceqzs_f32(float32_t a) {
  return (uint32_t)vceqzs_f32(a);
}

// COMMON-LABEL: test_vceqzd_f64
// UNCONSTRAINED: [[TMP0:%.*]] = fcmp oeq double %a, 0.000000e+00
// CONSTRAINED:   [[TMP0:%.*]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double 0.000000e+00, metadata !"oeq", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmp d{{[0-9]+}}, #0.0
// CHECK-ASM-NEXT:cset {{w[0-9]+}}, eq
// COMMONIR:      [[VCEQZ_I:%.*]] = sext i1 [[TMP0]] to i64
// COMMONIR:      ret i64 [[VCEQZ_I]]
uint64_t test_vceqzd_f64(float64_t a) {
  return (uint64_t)vceqzd_f64(a);
}

// COMMON-LABEL: test_vcges_f32
// UNCONSTRAINED: [[TMP0:%.*]] = fcmp oge float %a, %b
// CONSTRAINED:   [[TMP0:%.*]] = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"oge", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmp s{{[0-9]+}}, s{{[0-9]+}}
// CHECK-ASM-NEXT:cset {{w[0-9]+}}, ge
// COMMONIR:      [[VCMPD_I:%.*]] = sext i1 [[TMP0]] to i32
// COMMONIR:      ret i32 [[VCMPD_I]]
uint32_t test_vcges_f32(float32_t a, float32_t b) {
  return (uint32_t)vcges_f32(a, b);
}

// COMMON-LABEL: test_vcged_f64
// UNCONSTRAINED: [[TMP0:%.*]] = fcmp oge double %a, %b
// CONSTRAINED:   [[TMP0:%.*]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"oge", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmp d{{[0-9]+}}, d{{[0-9]+}}
// CHECK-ASM-NEXT:cset {{w[0-9]+}}, ge
// COMMONIR:      [[VCMPD_I:%.*]] = sext i1 [[TMP0]] to i64
// COMMONIR:      ret i64 [[VCMPD_I]]
uint64_t test_vcged_f64(float64_t a, float64_t b) {
  return (uint64_t)vcged_f64(a, b);
}

// COMMON-LABEL: test_vcgezs_f32
// UNCONSTRAINED: [[TMP0:%.*]] = fcmp oge float %a, 0.000000e+00
// CONSTRAINED:   [[TMP0:%.*]] = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float 0.000000e+00, metadata !"oge", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmp s{{[0-9]+}}, #0.0
// CHECK-ASM-NEXT:cset {{w[0-9]+}}, ge
// COMMONIR:      [[VCGEZ_I:%.*]] = sext i1 [[TMP0]] to i32
// COMMONIR:      ret i32 [[VCGEZ_I]]
uint32_t test_vcgezs_f32(float32_t a) {
  return (uint32_t)vcgezs_f32(a);
}

// COMMON-LABEL: test_vcgezd_f64
// UNCONSTRAINED: [[TMP0:%.*]] = fcmp oge double %a, 0.000000e+00
// CONSTRAINED:   [[TMP0:%.*]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double 0.000000e+00, metadata !"oge", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmp d{{[0-9]+}}, #0.0
// CHECK-ASM-NEXT:cset {{w[0-9]+}}, ge
// COMMONIR:      [[VCGEZ_I:%.*]] = sext i1 [[TMP0]] to i64
// COMMONIR:      ret i64 [[VCGEZ_I]]
uint64_t test_vcgezd_f64(float64_t a) {
  return (uint64_t)vcgezd_f64(a);
}

// COMMON-LABEL: test_vcgts_f32
// UNCONSTRAINED: [[TMP0:%.*]] = fcmp ogt float %a, %b
// CONSTRAINED:   [[TMP0:%.*]] = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"ogt", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmp s{{[0-9]+}}, s{{[0-9]+}}
// CHECK-ASM-NEXT:cset {{w[0-9]+}}, gt
// COMMONIR:      [[VCMPD_I:%.*]] = sext i1 [[TMP0]] to i32
// COMMONIR:      ret i32 [[VCMPD_I]]
uint32_t test_vcgts_f32(float32_t a, float32_t b) {
  return (uint32_t)vcgts_f32(a, b);
}

// COMMON-LABEL: test_vcgtd_f64
// UNCONSTRAINED: [[TMP0:%.*]] = fcmp ogt double %a, %b
// CONSTRAINED:   [[TMP0:%.*]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"ogt", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmp d{{[0-9]+}}, d{{[0-9]+}}
// CHECK-ASM-NEXT:cset {{w[0-9]+}}, gt
// COMMONIR:      [[VCMPD_I:%.*]] = sext i1 [[TMP0]] to i64
// COMMONIR:      ret i64 [[VCMPD_I]]
uint64_t test_vcgtd_f64(float64_t a, float64_t b) {
  return (uint64_t)vcgtd_f64(a, b);
}

// COMMON-LABEL: test_vcgtzs_f32
// UNCONSTRAINED: [[TMP0:%.*]] = fcmp ogt float %a, 0.000000e+00
// CONSTRAINED:   [[TMP0:%.*]] = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float 0.000000e+00, metadata !"ogt", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmp s{{[0-9]+}}, #0.0
// CHECK-ASM-NEXT:cset {{w[0-9]+}}, gt
// COMMONIR:      [[VCGTZ_I:%.*]] = sext i1 [[TMP0]] to i32
// COMMONIR:      ret i32 [[VCGTZ_I]]
uint32_t test_vcgtzs_f32(float32_t a) {
  return (uint32_t)vcgtzs_f32(a);
}

// COMMON-LABEL: test_vcgtzd_f64
// UNCONSTRAINED: [[TMP0:%.*]] = fcmp ogt double %a, 0.000000e+00
// CONSTRAINED:   [[TMP0:%.*]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double 0.000000e+00, metadata !"ogt", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmp d{{[0-9]+}}, #0.0
// CHECK-ASM-NEXT:cset {{w[0-9]+}}, gt
// COMMONIR:      [[VCGTZ_I:%.*]] = sext i1 [[TMP0]] to i64
// COMMONIR:      ret i64 [[VCGTZ_I]]
uint64_t test_vcgtzd_f64(float64_t a) {
  return (uint64_t)vcgtzd_f64(a);
}

// COMMON-LABEL: test_vcles_f32
// UNCONSTRAINED: [[TMP0:%.*]] = fcmp ole float %a, %b
// CONSTRAINED:   [[TMP0:%.*]] = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"ole", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmp s{{[0-9]+}}, s{{[0-9]+}}
// CHECK-ASM-NEXT:cset {{w[0-9]+}}, ls
// COMMONIR:      [[VCMPD_I:%.*]] = sext i1 [[TMP0]] to i32
// COMMONIR:      ret i32 [[VCMPD_I]]
uint32_t test_vcles_f32(float32_t a, float32_t b) {
  return (uint32_t)vcles_f32(a, b);
}

// COMMON-LABEL: test_vcled_f64
// UNCONSTRAINED: [[TMP0:%.*]] = fcmp ole double %a, %b
// CONSTRAINED:   [[TMP0:%.*]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"ole", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmp d{{[0-9]+}}, d{{[0-9]+}}
// CHECK-ASM-NEXT:cset {{w[0-9]+}}, ls
// COMMONIR:      [[VCMPD_I:%.*]] = sext i1 [[TMP0]] to i64
// COMMONIR:      ret i64 [[VCMPD_I]]
uint64_t test_vcled_f64(float64_t a, float64_t b) {
  return (uint64_t)vcled_f64(a, b);
}

// COMMON-LABEL: test_vclezs_f32
// UNCONSTRAINED: [[TMP0:%.*]] = fcmp ole float %a, 0.000000e+00
// CONSTRAINED:   [[TMP0:%.*]] = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float 0.000000e+00, metadata !"ole", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmp s{{[0-9]+}}, #0.0
// CHECK-ASM-NEXT:cset {{w[0-9]+}}, ls
// COMMONIR:      [[VCLEZ_I:%.*]] = sext i1 [[TMP0]] to i32
// COMMONIR:      ret i32 [[VCLEZ_I]]
uint32_t test_vclezs_f32(float32_t a) {
  return (uint32_t)vclezs_f32(a);
}

// COMMON-LABEL: test_vclezd_f64
// UNCONSTRAINED: [[TMP0:%.*]] = fcmp ole double %a, 0.000000e+00
// CONSTRAINED:   [[TMP0:%.*]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double 0.000000e+00, metadata !"ole", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmp d{{[0-9]+}}, #0.0
// CHECK-ASM-NEXT:cset {{w[0-9]+}}, ls
// COMMONIR:      [[VCLEZ_I:%.*]] = sext i1 [[TMP0]] to i64
// COMMONIR:      ret i64 [[VCLEZ_I]]
uint64_t test_vclezd_f64(float64_t a) {
  return (uint64_t)vclezd_f64(a);
}

// COMMON-LABEL: test_vclts_f32
// UNCONSTRAINED: [[TMP0:%.*]] = fcmp olt float %a, %b
// CONSTRAINED:   [[TMP0:%.*]] = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %b, metadata !"olt", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmp s{{[0-9]+}}, s{{[0-9]+}}
// CHECK-ASM-NEXT:cset {{w[0-9]+}}, mi
// COMMONIR:      [[VCMPD_I:%.*]] = sext i1 [[TMP0]] to i32
// COMMONIR:      ret i32 [[VCMPD_I]]
uint32_t test_vclts_f32(float32_t a, float32_t b) {
  return (uint32_t)vclts_f32(a, b);
}

// COMMON-LABEL: test_vcltd_f64
// UNCONSTRAINED: [[TMP0:%.*]] = fcmp olt double %a, %b
// CONSTRAINED:   [[TMP0:%.*]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double %b, metadata !"olt", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmp d{{[0-9]+}}, d{{[0-9]+}}
// CHECK-ASM-NEXT:cset {{w[0-9]+}}, mi
// COMMONIR:      [[VCMPD_I:%.*]] = sext i1 [[TMP0]] to i64
// COMMONIR:      ret i64 [[VCMPD_I]]
uint64_t test_vcltd_f64(float64_t a, float64_t b) {
  return (uint64_t)vcltd_f64(a, b);
}

// COMMON-LABEL: test_vcltzs_f32
// UNCONSTRAINED: [[TMP0:%.*]] = fcmp olt float %a, 0.000000e+00
// CONSTRAINED:   [[TMP0:%.*]] = call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float 0.000000e+00, metadata !"olt", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmp s{{[0-9]+}}, #0.0
// CHECK-ASM-NEXT:cset {{w[0-9]+}}, mi
// COMMONIR:      [[VCLTZ_I:%.*]] = sext i1 [[TMP0]] to i32
// COMMONIR:      ret i32 [[VCLTZ_I]]
uint32_t test_vcltzs_f32(float32_t a) {
  return (uint32_t)vcltzs_f32(a);
}

// COMMON-LABEL: test_vcltzd_f64
// UNCONSTRAINED: [[TMP0:%.*]] = fcmp olt double %a, 0.000000e+00
// CONSTRAINED:   [[TMP0:%.*]] = call i1 @llvm.experimental.constrained.fcmp.f64(double %a, double 0.000000e+00, metadata !"olt", metadata !"fpexcept.strict")
// CHECK-ASM:     fcmp d{{[0-9]+}}, #0.0
// CHECK-ASM-NEXT:cset {{w[0-9]+}}, mi
// COMMONIR:      [[VCLTZ_I:%.*]] = sext i1 [[TMP0]] to i64
// COMMONIR:      ret i64 [[VCLTZ_I]]
uint64_t test_vcltzd_f64(float64_t a) {
  return (uint64_t)vcltzd_f64(a);
}

// COMMON-LABEL: test_vadd_f64
// UNCONSTRAINED: [[ADD_I:%.*]] = fadd <1 x double> %a, %b
// CONSTRAINED:   [[ADD_I:%.*]] = call <1 x double> @llvm.experimental.constrained.fadd.v1f64(<1 x double> %a, <1 x double> %b, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fadd d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
// COMMONIR:      ret <1 x double> [[ADD_I]]
float64x1_t test_vadd_f64(float64x1_t a, float64x1_t b) {
  return vadd_f64(a, b);
}

// COMMON-LABEL: test_vmul_f64
// UNCONSTRAINED: [[MUL_I:%.*]] = fmul <1 x double> %a, %b
// CONSTRAINED:   [[MUL_I:%.*]] = call <1 x double> @llvm.experimental.constrained.fmul.v1f64(<1 x double> %a, <1 x double> %b, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fmul d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
// COMMONIR:      ret <1 x double> [[MUL_I]]
float64x1_t test_vmul_f64(float64x1_t a, float64x1_t b) {
  return vmul_f64(a, b);
}

// COMMON-LABEL: test_vdiv_f64
// UNCONSTRAINED: [[DIV_I:%.*]] = fdiv <1 x double> %a, %b
// CONSTRAINED:   [[DIV_I:%.*]] = call <1 x double> @llvm.experimental.constrained.fdiv.v1f64(<1 x double> %a, <1 x double> %b, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fdiv d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
// COMMONIR:      ret <1 x double> [[DIV_I]]
float64x1_t test_vdiv_f64(float64x1_t a, float64x1_t b) {
  return vdiv_f64(a, b);
}

// COMMON-LABEL: test_vmla_f64
// UNCONSTRAINED: [[MUL_I:%.*]] = fmul <1 x double> %b, %c
// CONSTRAINED:   [[MUL_I:%.*]] = call <1 x double> @llvm.experimental.constrained.fmul.v1f64(<1 x double> %b, <1 x double> %c, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fmul d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
// UNCONSTRAINED: [[ADD_I:%.*]] = fadd <1 x double> %a, [[MUL_I]]
// CONSTRAINED:   [[ADD_I:%.*]] = call <1 x double> @llvm.experimental.constrained.fadd.v1f64(<1 x double> %a, <1 x double> [[MUL_I]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fadd d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
// COMMONIR:      ret <1 x double> [[ADD_I]]
float64x1_t test_vmla_f64(float64x1_t a, float64x1_t b, float64x1_t c) {
  return vmla_f64(a, b, c);
}

// COMMON-LABEL: test_vmls_f64
// UNCONSTRAINED: [[MUL_I:%.*]] = fmul <1 x double> %b, %c
// CONSTRAINED:   [[MUL_I:%.*]] = call <1 x double> @llvm.experimental.constrained.fmul.v1f64(<1 x double> %b, <1 x double> %c, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fmul d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
// UNCONSTRAINED: [[SUB_I:%.*]] = fsub <1 x double> %a, [[MUL_I]]
// CONSTRAINED:   [[SUB_I:%.*]] = call <1 x double> @llvm.experimental.constrained.fsub.v1f64(<1 x double> %a, <1 x double> [[MUL_I]], metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fsub d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
// COMMONIR:      ret <1 x double> [[SUB_I]]
float64x1_t test_vmls_f64(float64x1_t a, float64x1_t b, float64x1_t c) {
  return vmls_f64(a, b, c);
}

// COMMON-LABEL: test_vfma_f64
// COMMONIR:      [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// COMMONIR:      [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// COMMONIR:      [[TMP2:%.*]] = bitcast <1 x double> %c to <8 x i8>
// UNCONSTRAINED: [[TMP3:%.*]] = call <1 x double> @llvm.fma.v1f64(<1 x double> %b, <1 x double> %c, <1 x double> %a)
// CONSTRAINED:   [[TMP3:%.*]] = call <1 x double> @llvm.experimental.constrained.fma.v1f64(<1 x double> %b, <1 x double> %c, <1 x double> %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fmadd d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
// COMMONIR:      ret <1 x double> [[TMP3]]
float64x1_t test_vfma_f64(float64x1_t a, float64x1_t b, float64x1_t c) {
  return vfma_f64(a, b, c);
}

// COMMON-LABEL: test_vfms_f64
// COMMONIR:      [[SUB_I:%.*]] = fneg <1 x double> %b
// CHECK-ASM:     fneg d{{[0-9]+}}, d{{[0-9]+}}
// COMMONIR:      [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// COMMONIR:      [[TMP1:%.*]] = bitcast <1 x double> [[SUB_I]] to <8 x i8>
// COMMONIR:      [[TMP2:%.*]] = bitcast <1 x double> %c to <8 x i8>
// UNCONSTRAINED: [[TMP3:%.*]] = call <1 x double> @llvm.fma.v1f64(<1 x double> [[SUB_I]], <1 x double> %c, <1 x double> %a)
// CONSTRAINED:   [[TMP3:%.*]] = call <1 x double> @llvm.experimental.constrained.fma.v1f64(<1 x double> [[SUB_I]], <1 x double> %c, <1 x double> %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fmadd d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
// COMMONIR:      ret <1 x double> [[TMP3]]
float64x1_t test_vfms_f64(float64x1_t a, float64x1_t b, float64x1_t c) {
  return vfms_f64(a, b, c);
}

// COMMON-LABEL: test_vsub_f64
// UNCONSTRAINED: [[SUB_I:%.*]] = fsub <1 x double> %a, %b
// CONSTRAINED:   [[SUB_I:%.*]] = call <1 x double> @llvm.experimental.constrained.fsub.v1f64(<1 x double> %a, <1 x double> %b, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fsub d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
// COMMONIR:      ret <1 x double> [[SUB_I]]
float64x1_t test_vsub_f64(float64x1_t a, float64x1_t b) {
  return vsub_f64(a, b);
}

// COMMON-LABEL: test_vcvt_s64_f64
// COMMONIR:      [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// UNCONSTRAINED: [[TMP1:%.*]] = fptosi <1 x double> %a to <1 x i64>
// CONSTRAINED:   [[TMP1:%.*]] = call <1 x i64> @llvm.experimental.constrained.fptosi.v1i64.v1f64(<1 x double> %a, metadata !"fpexcept.strict")
// CHECK-ASM:     fcvtzs x{{[0-9]+}}, d{{[0-9]+}}
// COMMONIR:      ret <1 x i64> [[TMP1]]
int64x1_t test_vcvt_s64_f64(float64x1_t a) {
  return vcvt_s64_f64(a);
}

// COMMON-LABEL: test_vcvt_u64_f64
// COMMONIR:      [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// UNCONSTRAINED: [[TMP1:%.*]] = fptoui <1 x double> %a to <1 x i64>
// CONSTRAINED:   [[TMP1:%.*]] = call <1 x i64> @llvm.experimental.constrained.fptoui.v1i64.v1f64(<1 x double> %a, metadata !"fpexcept.strict")
// CHECK-ASM:     fcvtzu x{{[0-9]+}}, d{{[0-9]+}}
// COMMONIR:      ret <1 x i64> [[TMP1]]
uint64x1_t test_vcvt_u64_f64(float64x1_t a) {
  return vcvt_u64_f64(a);
}

// COMMON-LABEL: test_vcvt_f64_s64
// COMMONIR:      [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// UNCONSTRAINED: [[VCVT_I:%.*]] = sitofp <1 x i64> %a to <1 x double>
// CONSTRAINED:   [[VCVT_I:%.*]] = call <1 x double> @llvm.experimental.constrained.sitofp.v1f64.v1i64(<1 x i64> %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     scvtf d{{[0-9]+}}, x{{[0-9]+}}
// COMMONIR:      ret <1 x double> [[VCVT_I]]
float64x1_t test_vcvt_f64_s64(int64x1_t a) {
  return vcvt_f64_s64(a);
}

// COMMON-LABEL: test_vcvt_f64_u64
// COMMONIR:      [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// UNCONSTRAINED: [[VCVT_I:%.*]] = uitofp <1 x i64> %a to <1 x double>
// CONSTRAINED:   [[VCVT_I:%.*]] = call <1 x double> @llvm.experimental.constrained.uitofp.v1f64.v1i64(<1 x i64> %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     ucvtf d{{[0-9]+}}, x{{[0-9]+}}
// COMMONIR:      ret <1 x double> [[VCVT_I]]
float64x1_t test_vcvt_f64_u64(uint64x1_t a) {
  return vcvt_f64_u64(a);
}

// COMMON-LABEL: test_vrnda_f64
// COMMONIR:      [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// UNCONSTRAINED: [[VRNDA1_I:%.*]] = call <1 x double> @llvm.round.v1f64(<1 x double> %a)
// CONSTRAINED:   [[VRNDA1_I:%.*]] = call <1 x double> @llvm.experimental.constrained.round.v1f64(<1 x double> %a, metadata !"fpexcept.strict")
// CHECK-ASM:     frinta d{{[0-9]+}}, d{{[0-9]+}}
// COMMONIR:      ret <1 x double> [[VRNDA1_I]]
float64x1_t test_vrnda_f64(float64x1_t a) {
  return vrnda_f64(a);
}

// COMMON-LABEL: test_vrndp_f64
// COMMONIR:      [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// UNCONSTRAINED: [[VRNDP1_I:%.*]] = call <1 x double> @llvm.ceil.v1f64(<1 x double> %a)
// CONSTRAINED:   [[VRNDP1_I:%.*]] = call <1 x double> @llvm.experimental.constrained.ceil.v1f64(<1 x double> %a, metadata !"fpexcept.strict")
// CHECK-ASM:     frintp d{{[0-9]+}}, d{{[0-9]+}}
// COMMONIR:      ret <1 x double> [[VRNDP1_I]]
float64x1_t test_vrndp_f64(float64x1_t a) {
  return vrndp_f64(a);
}

// COMMON-LABEL: test_vrndm_f64
// COMMONIR:      [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// UNCONSTRAINED: [[VRNDM1_I:%.*]] = call <1 x double> @llvm.floor.v1f64(<1 x double> %a)
// CONSTRAINED:   [[VRNDM1_I:%.*]] = call <1 x double> @llvm.experimental.constrained.floor.v1f64(<1 x double> %a, metadata !"fpexcept.strict")
// CHECK-ASM:     frintm d{{[0-9]+}}, d{{[0-9]+}}
// COMMONIR:      ret <1 x double> [[VRNDM1_I]]
float64x1_t test_vrndm_f64(float64x1_t a) {
  return vrndm_f64(a);
}

// COMMON-LABEL: test_vrndx_f64
// COMMONIR:      [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// UNCONSTRAINED: [[VRNDX1_I:%.*]] = call <1 x double> @llvm.rint.v1f64(<1 x double> %a)
// CONSTRAINED:   [[VRNDX1_I:%.*]] = call <1 x double> @llvm.experimental.constrained.rint.v1f64(<1 x double> %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     frintx d{{[0-9]+}}, d{{[0-9]+}}
// COMMONIR:      ret <1 x double> [[VRNDX1_I]]
float64x1_t test_vrndx_f64(float64x1_t a) {
  return vrndx_f64(a);
}

// COMMON-LABEL: test_vrnd_f64
// COMMONIR:      [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// UNCONSTRAINED: [[VRNDZ1_I:%.*]] = call <1 x double> @llvm.trunc.v1f64(<1 x double> %a)
// CONSTRAINED:   [[VRNDZ1_I:%.*]] = call <1 x double> @llvm.experimental.constrained.trunc.v1f64(<1 x double> %a, metadata !"fpexcept.strict")
// CHECK-ASM:     frintz d{{[0-9]+}}, d{{[0-9]+}}
// COMMONIR:      ret <1 x double> [[VRNDZ1_I]]
float64x1_t test_vrnd_f64(float64x1_t a) {
  return vrnd_f64(a);
}

// COMMON-LABEL: test_vrndi_f64
// COMMONIR:      [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// UNCONSTRAINED: [[VRNDI1_I:%.*]] = call <1 x double> @llvm.nearbyint.v1f64(<1 x double> %a)
// CONSTRAINED:   [[VRNDI1_I:%.*]] = call <1 x double> @llvm.experimental.constrained.nearbyint.v1f64(<1 x double> %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     frinti d{{[0-9]+}}, d{{[0-9]+}}
// COMMONIR:      ret <1 x double> [[VRNDI1_I]]
float64x1_t test_vrndi_f64(float64x1_t a) {
  return vrndi_f64(a);
}

// COMMON-LABEL: test_vsqrt_f64
// COMMONIR:      [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// UNCONSTRAINED: [[VSQRT_I:%.*]] = call <1 x double> @llvm.sqrt.v1f64(<1 x double> %a)
// CONSTRAINED:   [[VSQRT_I:%.*]] = call <1 x double> @llvm.experimental.constrained.sqrt.v1f64(<1 x double> %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fsqrt d{{[0-9]+}}, d{{[0-9]+}}
// COMMONIR:      ret <1 x double> [[VSQRT_I]]
float64x1_t test_vsqrt_f64(float64x1_t a) {
  return vsqrt_f64(a);
}
