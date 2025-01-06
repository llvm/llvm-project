// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:  -disable-O0-optnone -emit-llvm -o - %s \
// RUN: | opt -S -passes=mem2reg | FileCheck --check-prefix=COMMON --check-prefix=COMMONIR --check-prefix=UNCONSTRAINED %s
// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:  -ffp-exception-behavior=strict \
// RUN:  -disable-O0-optnone -emit-llvm -o - %s \
// RUN: | opt -S -passes=mem2reg | FileCheck --check-prefix=COMMON --check-prefix=COMMONIR --check-prefix=CONSTRAINED %s
// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:  -disable-O0-optnone -S -o - %s \
// RUN: | FileCheck --check-prefix=COMMON --check-prefix=CHECK-ASM %s
// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:  -ffp-exception-behavior=strict \
// RUN:  -disable-O0-optnone -S -o - %s \
// RUN: | FileCheck --check-prefix=COMMON --check-prefix=CHECK-ASM %s

// REQUIRES: aarch64-registered-target

// Test new aarch64 intrinsics and types but constrained

#include <arm_neon.h>

// COMMON-LABEL: test_vrndaq_f64
// COMMONIR:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// UNCONSTRAINED: [[VRNDA1_I:%.*]] = call <2 x double> @llvm.round.v2f64(<2 x double> %a)
// CONSTRAINED:   [[VRNDA1_I:%.*]] = call <2 x double> @llvm.experimental.constrained.round.v2f64(<2 x double> %a, metadata !"fpexcept.strict")
// CHECK-ASM:     frinta v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
// COMMONIR:      ret <2 x double> [[VRNDA1_I]]
float64x2_t test_vrndaq_f64(float64x2_t a) {
  return vrndaq_f64(a);
}

// COMMON-LABEL: test_vrndpq_f64
// COMMONIR:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// UNCONSTRAINED: [[VRNDP1_I:%.*]] = call <2 x double> @llvm.ceil.v2f64(<2 x double> %a)
// CONSTRAINED:   [[VRNDP1_I:%.*]] = call <2 x double> @llvm.experimental.constrained.ceil.v2f64(<2 x double> %a, metadata !"fpexcept.strict")
// CHECK-ASM:     frintp v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
// COMMONIR:      ret <2 x double> [[VRNDP1_I]]
float64x2_t test_vrndpq_f64(float64x2_t a) {
  return vrndpq_f64(a);
}

// COMMON-LABEL: test_vsqrtq_f32
// COMMONIR:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// UNCONSTRAINED: [[VSQRT_I:%.*]] = call <4 x float> @llvm.sqrt.v4f32(<4 x float> %a)
// CONSTRAINED:   [[VSQRT_I:%.*]] = call <4 x float> @llvm.experimental.constrained.sqrt.v4f32(<4 x float> %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fsqrt v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
// COMMONIR:      ret <4 x float> [[VSQRT_I]]
float32x4_t test_vsqrtq_f32(float32x4_t a) {
  return vsqrtq_f32(a);
}

// COMMON-LABEL: test_vsqrtq_f64
// COMMONIR:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// UNCONSTRAINED: [[VSQRT_I:%.*]] = call <2 x double> @llvm.sqrt.v2f64(<2 x double> %a)
// CONSTRAINED:   [[VSQRT_I:%.*]] = call <2 x double> @llvm.experimental.constrained.sqrt.v2f64(<2 x double> %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
// CHECK-ASM:     fsqrt v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
// COMMONIR:      ret <2 x double> [[VSQRT_I]]
float64x2_t test_vsqrtq_f64(float64x2_t a) {
  return vsqrtq_f64(a);
}
