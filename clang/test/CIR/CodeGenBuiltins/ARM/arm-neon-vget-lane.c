// vget_lane/vgetq_lane lower to __builtin_neon_* on 32-bit ARM, unlike AArch64.
// CIR and classic CodeGen must produce the same extract, so both feed LLVM.

// RUN: %clang_cc1 -triple armv7-unknown-linux-gnueabihf -target-feature +neon -target-feature +bf16 -ffreestanding -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple armv7-unknown-linux-gnueabihf -target-feature +neon -target-feature +bf16 -ffreestanding -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple armv7-unknown-linux-gnueabihf -target-feature +neon -target-feature +bf16 -ffreestanding -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

#include <arm_neon.h>

// CIR-LABEL: cir.func{{.*}} @get_s32(
// CIR: cir.vec.extract {{.*}} : !cir.vector<4 x !s32i>
// LLVM-LABEL: define{{.*}} @get_s32(
// LLVM: extractelement <4 x i32> %{{.*}}, i32 2
int get_s32(int32x4_t v) { return vgetq_lane_s32(v, 2); }

// CIR-LABEL: cir.func{{.*}} @get_f32(
// CIR: cir.vec.extract {{.*}} : !cir.vector<4 x !cir.float>
// LLVM-LABEL: define{{.*}} @get_f32(
// LLVM: extractelement <4 x float> %{{.*}}, i32 1
float get_f32(float32x4_t v) { return vgetq_lane_f32(v, 1); }

// CIR-LABEL: cir.func{{.*}} @get_s16(
// CIR: cir.vec.extract {{.*}} : !cir.vector<4 x !s16i>
// LLVM-LABEL: define{{.*}} @get_s16(
// LLVM: extractelement <4 x i16> %{{.*}}, i32 3
short get_s16(int16x4_t v) { return vget_lane_s16(v, 3); }

// A 64-bit lane, and the single-element vector shape.
// CIR-LABEL: cir.func{{.*}} @get_s64(
// CIR: cir.vec.extract {{.*}} : !cir.vector<1 x !s64i>
// LLVM-LABEL: define{{.*}} @get_s64(
// LLVM: extractelement <1 x i64> %{{.*}}, i32 0
long long get_s64(int64x1_t v) { return vget_lane_s64(v, 0); }

// The header bitcasts the unsigned vector to the signed builtin type, so the
// extract is on !s8i even though the parameter is !u8i.
// CIR-LABEL: cir.func{{.*}} @get_u8(
// CIR: cir.vec.extract {{.*}} : !cir.vector<8 x !s8i>
// LLVM-LABEL: define{{.*}} @get_u8(
// LLVM: extractelement <8 x i8> %{{.*}}, i32 5
unsigned char get_u8(uint8x8_t v) { return vget_lane_u8(v, 5); }

// CIR-LABEL: cir.func{{.*}} @get_bf16(
// CIR: cir.vec.extract {{.*}} : !cir.vector<4 x !cir.bf16>
// LLVM-LABEL: define{{.*}} @get_bf16(
// LLVM: extractelement <4 x bfloat> %{{.*}}, i32 1
bfloat16_t get_bf16(bfloat16x4_t v) { return vget_lane_bf16(v, 1); }
