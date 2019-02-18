// RUN: %clang_cc1 -triple arm64-apple-ios7 -target-feature +neon -ffreestanding -emit-llvm -o - %s | FileCheck %s

#include <arm_neon.h>

int64x2_t rnd5(float64x2_t a) { return vrndq_f64(a); }
// CHECK: call <2 x double> @llvm.trunc.v2f64(<2 x double>

int64x2_t rnd9(float64x2_t a) { return vrndnq_f64(a); }
// CHECK: call <2 x double> @llvm.aarch64.neon.frintn.v2f64(<2 x double>

int64x2_t rnd13(float64x2_t a) { return vrndmq_f64(a); }
// CHECK: call <2 x double> @llvm.floor.v2f64(<2 x double>

int64x2_t rnd18(float64x2_t a) { return vrndpq_f64(a); }
// CHECK: call <2 x double> @llvm.ceil.v2f64(<2 x double>

int64x2_t rnd22(float64x2_t a) { return vrndaq_f64(a); }
// CHECK: call <2 x double> @llvm.round.v2f64(<2 x double>

int64x2_t rnd25(float64x2_t a) { return vrndxq_f64(a); }
// CHECK: call <2 x double> @llvm.rint.v2f64(<2 x double>

