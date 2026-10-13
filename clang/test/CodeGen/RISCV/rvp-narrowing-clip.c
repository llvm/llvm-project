// RUN: %clang_cc1 -triple riscv32 -target-feature +experimental-p \
// RUN:   -disable-O0-optnone -emit-llvm -o - %s \
// RUN:   | opt -S -passes=sroa,instcombine | FileCheck %s --check-prefixes=CHECK,RV32
// RUN: %clang_cc1 -triple riscv64 -target-feature +experimental-p \
// RUN:   -disable-O0-optnone -emit-llvm -o - %s \
// RUN:   | opt -S -passes=sroa,instcombine | FileCheck %s --check-prefixes=CHECK,RV64

#include <riscv_packed_simd.h>

// CHECK-LABEL: define dso_local i32 @test_pnclipu_s_u8x4(
// RV32-SAME: i64 noundef {{.*}}, i32 noundef [[SHAMT:%.*]])
// RV64-SAME: i64 noundef {{.*}}, i32 noundef signext [[SHAMT:%.*]])
// CHECK: call <4 x i8> @llvm.riscv.pnclipu.v4i8.v4i16(<4 x i16> {{.*}}, i32 [[SHAMT]])
uint8x4_t test_pnclipu_s_u8x4(uint16x4_t rs1, unsigned shamt) {
  return __riscv_pnclipu_s_u8x4(rs1, shamt);
}

// CHECK-LABEL: define dso_local i32 @test_pnclipu_s_u16x2(
// RV32-SAME: i64 noundef {{.*}}, i32 noundef [[SHAMT:%.*]])
// RV64-SAME: i64 noundef {{.*}}, i32 noundef signext [[SHAMT:%.*]])
// CHECK: call <2 x i16> @llvm.riscv.pnclipu.v2i16.v2i32(<2 x i32> {{.*}}, i32 [[SHAMT]])
uint16x2_t test_pnclipu_s_u16x2(uint32x2_t rs1, unsigned shamt) {
  return __riscv_pnclipu_s_u16x2(rs1, shamt);
}

// CHECK-LABEL: define dso_local i32 @test_pnclipru_s_u8x4(
// RV32-SAME: i64 noundef {{.*}}, i32 noundef [[SHAMT:%.*]])
// RV64-SAME: i64 noundef {{.*}}, i32 noundef signext [[SHAMT:%.*]])
// CHECK: call <4 x i8> @llvm.riscv.pnclipru.v4i8.v4i16(<4 x i16> {{.*}}, i32 [[SHAMT]])
uint8x4_t test_pnclipru_s_u8x4(uint16x4_t rs1, unsigned shamt) {
  return __riscv_pnclipru_s_u8x4(rs1, shamt);
}

// CHECK-LABEL: define dso_local i32 @test_pnclipru_s_u16x2(
// RV32-SAME: i64 noundef {{.*}}, i32 noundef [[SHAMT:%.*]])
// RV64-SAME: i64 noundef {{.*}}, i32 noundef signext [[SHAMT:%.*]])
// CHECK: call <2 x i16> @llvm.riscv.pnclipru.v2i16.v2i32(<2 x i32> {{.*}}, i32 [[SHAMT]])
uint16x2_t test_pnclipru_s_u16x2(uint32x2_t rs1, unsigned shamt) {
  return __riscv_pnclipru_s_u16x2(rs1, shamt);
}

// CHECK-LABEL: define dso_local i32 @test_pnclip_s_i8x4(
// RV32-SAME: i64 noundef {{.*}}, i32 noundef [[SHAMT:%.*]])
// RV64-SAME: i64 noundef {{.*}}, i32 noundef signext [[SHAMT:%.*]])
// CHECK: call <4 x i8> @llvm.riscv.pnclip.v4i8.v4i16(<4 x i16> {{.*}}, i32 [[SHAMT]])
int8x4_t test_pnclip_s_i8x4(int16x4_t rs1, unsigned shamt) {
  return __riscv_pnclip_s_i8x4(rs1, shamt);
}

// CHECK-LABEL: define dso_local i32 @test_pnclip_s_i16x2(
// RV32-SAME: i64 noundef {{.*}}, i32 noundef [[SHAMT:%.*]])
// RV64-SAME: i64 noundef {{.*}}, i32 noundef signext [[SHAMT:%.*]])
// CHECK: call <2 x i16> @llvm.riscv.pnclip.v2i16.v2i32(<2 x i32> {{.*}}, i32 [[SHAMT]])
int16x2_t test_pnclip_s_i16x2(int32x2_t rs1, unsigned shamt) {
  return __riscv_pnclip_s_i16x2(rs1, shamt);
}

// CHECK-LABEL: define dso_local i32 @test_pnclipr_s_i8x4(
// RV32-SAME: i64 noundef {{.*}}, i32 noundef [[SHAMT:%.*]])
// RV64-SAME: i64 noundef {{.*}}, i32 noundef signext [[SHAMT:%.*]])
// CHECK: call <4 x i8> @llvm.riscv.pnclipr.v4i8.v4i16(<4 x i16> {{.*}}, i32 [[SHAMT]])
int8x4_t test_pnclipr_s_i8x4(int16x4_t rs1, unsigned shamt) {
  return __riscv_pnclipr_s_i8x4(rs1, shamt);
}

// CHECK-LABEL: define dso_local i32 @test_pnclipr_s_i16x2(
// RV32-SAME: i64 noundef {{.*}}, i32 noundef [[SHAMT:%.*]])
// RV64-SAME: i64 noundef {{.*}}, i32 noundef signext [[SHAMT:%.*]])
// CHECK: call <2 x i16> @llvm.riscv.pnclipr.v2i16.v2i32(<2 x i32> {{.*}}, i32 [[SHAMT]])
int16x2_t test_pnclipr_s_i16x2(int32x2_t rs1, unsigned shamt) {
  return __riscv_pnclipr_s_i16x2(rs1, shamt);
}
