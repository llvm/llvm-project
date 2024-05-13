// RUN: %clang_cc1 -triple arm64-apple-ios7 -target-feature +neon -ffreestanding -flax-vector-conversions=none -emit-llvm -o - %s \
// RUN: | FileCheck --check-prefix=COMMON --check-prefix=UNCONSTRAINED %s
// RUN: %clang_cc1 -triple arm64-apple-ios7 -target-feature +neon -ffreestanding -flax-vector-conversions=none -ffp-exception-behavior=strict -emit-llvm -o - %s \
// RUN: | FileCheck --check-prefix=COMMON --check-prefix=CONSTRAINED %s
// RUN: %clang_cc1 -triple arm64-apple-ios7 -target-feature +neon -ffreestanding -flax-vector-conversions=none -emit-llvm -o - %s | llc -o=- - \
// RUN: | FileCheck --check-prefix=COMMON --check-prefix=CHECK-ASM %s
// RUN: %clang_cc1 -triple arm64-apple-ios7 -target-feature +neon -ffreestanding -flax-vector-conversions=none -ffp-exception-behavior=strict -emit-llvm -o - %s | llc -o=- - \
// RUN: | FileCheck --check-prefix=COMMON --check-prefix=CHECK-ASM %s

// REQUIRES: aarch64-registered-target

#include <arm_neon.h>

float64x2_t rnd5(float64x2_t a) { return vrndq_f64(a); }
// COMMON-LABEL: rnd5
// UNCONSTRAINED: call <2 x double> @llvm.trunc.v2f64(<2 x double>
// CONSTRAINED:   call <2 x double> @llvm.experimental.constrained.trunc.v2f64(<2 x double>
// CHECK-ASM:     frintz.2d v{{[0-9]+}}, v{{[0-9]+}}

float64x2_t rnd13(float64x2_t a) { return vrndmq_f64(a); }
// COMMON-LABEL: rnd13
// UNCONSTRAINED: call <2 x double> @llvm.floor.v2f64(<2 x double>
// CONSTRAINED:   call <2 x double> @llvm.experimental.constrained.floor.v2f64(<2 x double>
// CHECK-ASM:     frintm.2d v{{[0-9]+}}, v{{[0-9]+}}

float64x2_t rnd18(float64x2_t a) { return vrndpq_f64(a); }
// COMMON-LABEL: rnd18
// UNCONSTRAINED: call <2 x double> @llvm.ceil.v2f64(<2 x double>
// CONSTRAINED:   call <2 x double> @llvm.experimental.constrained.ceil.v2f64(<2 x double>
// CHECK-ASM:     frintp.2d v{{[0-9]+}}, v{{[0-9]+}}

float64x2_t rnd22(float64x2_t a) { return vrndaq_f64(a); }
// COMMON-LABEL: rnd22
// UNCONSTRAINED: call <2 x double> @llvm.round.v2f64(<2 x double>
// CONSTRAINED:   call <2 x double> @llvm.experimental.constrained.round.v2f64(<2 x double>
// CHECK-ASM:     frinta.2d v{{[0-9]+}}, v{{[0-9]+}}

float64x2_t rnd25(float64x2_t a) { return vrndxq_f64(a); }
// COMMON-LABEL: rnd25
// UNCONSTRAINED: call <2 x double> @llvm.rint.v2f64(<2 x double>
// CONSTRAINED:   call <2 x double> @llvm.experimental.constrained.rint.v2f64(<2 x double>
// CHECK-ASM:     frintx.2d v{{[0-9]+}}, v{{[0-9]+}}

