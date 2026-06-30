// On 32-bit ARM the NEON vget_lane/vgetq_lane intrinsics lower to
// __builtin_neon_* (unlike AArch64); check CIR lowers them to a vector extract.

// RUN: %clang_cc1 -triple armv7-unknown-linux-gnueabihf -target-feature +neon -ffreestanding -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple armv7-unknown-linux-gnueabihf -target-feature +neon -ffreestanding -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

#include <arm_neon.h>

// CIR-LABEL: cir.func{{.*}} @get_s32(
// CIR: cir.vec.extract {{.*}} : !cir.vector<4 x !s32i>
// LLVM-LABEL: define dso_local i32 @get_s32(
// LLVM: extractelement <4 x i32> %{{.*}}, i32 2
int get_s32(int32x4_t v) { return vgetq_lane_s32(v, 2); }

// CIR-LABEL: cir.func{{.*}} @get_f32(
// CIR: cir.vec.extract {{.*}} : !cir.vector<4 x !cir.float>
// LLVM-LABEL: define dso_local float @get_f32(
// LLVM: extractelement <4 x float> %{{.*}}, i32 1
float get_f32(float32x4_t v) { return vgetq_lane_f32(v, 1); }

// CIR-LABEL: cir.func{{.*}} @get_s16(
// CIR: cir.vec.extract {{.*}} : !cir.vector<4 x !s16i>
// LLVM-LABEL: define dso_local i16 @get_s16(
// LLVM: extractelement <4 x i16> %{{.*}}, i32 3
short get_s16(int16x4_t v) { return vget_lane_s16(v, 3); }
