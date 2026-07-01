// RUN: %clang_cc1 -triple riscv32 -target-feature +xcvalu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple riscv32 -target-feature +xcvalu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple riscv32 -target-feature +xcvalu -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

#include <stdint.h>

// CIR-LABEL: @test_alu_sle(
// CIR: [[CMP:%.*]] = cir.cmp le {{%.*}}, {{%.*}} : !s32i
// CIR: cir.cast bool_to_int [[CMP]] : !cir.bool -> !s32i
// LLVM-LABEL: @test_alu_sle(
// LLVM: [[CMP:%.*]] = icmp sle i32 {{%.*}}, {{%.*}}
// LLVM: zext i1 [[CMP]] to i32
int test_alu_sle(int32_t a, int32_t b) {
  return __builtin_riscv_cv_alu_sle(a, b);
}

// CIR-LABEL: @test_alu_sleu(
// CIR: [[CMP:%.*]] = cir.cmp le {{%.*}}, {{%.*}} : !u32i
// CIR: cir.cast bool_to_int [[CMP]] : !cir.bool -> !s32i
// LLVM-LABEL: @test_alu_sleu(
// LLVM: [[CMP:%.*]] = icmp ule i32 {{%.*}}, {{%.*}}
// LLVM: zext i1 [[CMP]] to i32
int test_alu_sleu(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_alu_sleu(a, b);
}

// CIR-LABEL: @test_alu_exths(
// CIR: [[TRUNC:%.*]] = cir.cast integral {{%.*}} : !s32i -> !s16i
// CIR: cir.cast integral [[TRUNC]] : !s16i -> !s32i
// LLVM-LABEL: @test_alu_exths(
// LLVM: [[TRUNC:%.*]] = trunc i32 {{%.*}} to i16
// LLVM: sext i16 [[TRUNC]] to i32
int test_alu_exths(int16_t a) {
  return __builtin_riscv_cv_alu_exths(a);
}

// CIR-LABEL: @test_alu_exthz(
// CIR: [[TRUNC:%.*]] = cir.cast integral {{%.*}} : !u32i -> !u16i
// CIR: cir.cast integral [[TRUNC]] : !u16i -> !u32i
// LLVM-LABEL: @test_alu_exthz(
// LLVM: [[TRUNC:%.*]] = trunc i32 {{%.*}} to i16
// LLVM: zext i16 [[TRUNC]] to i32
unsigned int test_alu_exthz(uint16_t a) {
  return __builtin_riscv_cv_alu_exthz(a);
}

// CIR-LABEL: @test_alu_extbs(
// CIR: [[TRUNC:%.*]] = cir.cast integral {{%.*}} : !s32i -> !s8i
// CIR: cir.cast integral [[TRUNC]] : !s8i -> !s32i
// LLVM-LABEL: @test_alu_extbs(
// LLVM: [[TRUNC:%.*]] = trunc i32 {{%.*}} to i8
// LLVM: sext i8 [[TRUNC]] to i32
int test_alu_extbs(int8_t a) {
  return __builtin_riscv_cv_alu_extbs(a);
}

// CIR-LABEL: @test_alu_extbz(
// CIR: [[TRUNC:%.*]] = cir.cast integral {{%.*}} : !u32i -> !u8i
// CIR: cir.cast integral [[TRUNC]] : !u8i -> !u32i
// LLVM-LABEL: @test_alu_extbz(
// LLVM: [[TRUNC:%.*]] = trunc i32 {{%.*}} to i8
// LLVM: zext i8 [[TRUNC]] to i32
unsigned int test_alu_extbz(uint8_t a) {
  return __builtin_riscv_cv_alu_extbz(a);
}

// CIR-LABEL: @test_alu_clip(
// CIR: cir.call_llvm_intrinsic "riscv.cv.alu.clip" {{%.*}}, {{%.*}} : (!s32i, !s32i) -> !s32i
// LLVM-LABEL: @test_alu_clip(
// LLVM: call i32 @llvm.riscv.cv.alu.clip(i32 {{%.*}}, i32 15)
int test_alu_clip(int32_t a) {
  return __builtin_riscv_cv_alu_clip(a, 15);
}

// CIR-LABEL: @test_alu_clipu(
// CIR: cir.call_llvm_intrinsic "riscv.cv.alu.clipu" {{%.*}}, {{%.*}} : (!u32i, !u32i) -> !u32i
// LLVM-LABEL: @test_alu_clipu(
// LLVM: call i32 @llvm.riscv.cv.alu.clipu(i32 {{%.*}}, i32 15)
unsigned int test_alu_clipu(uint32_t a) {
  return __builtin_riscv_cv_alu_clipu(a, 15);
}

// CIR-LABEL: @test_alu_addN(
// CIR: cir.call_llvm_intrinsic "riscv.cv.alu.addN" {{%.*}}, {{%.*}}, {{%.*}} : (!s32i, !s32i, !u32i) -> !s32i
// LLVM-LABEL: @test_alu_addN(
// LLVM: call i32 @llvm.riscv.cv.alu.addN(i32 {{%.*}}, i32 {{%.*}}, i32 0)
int test_alu_addN(int32_t a, int32_t b) {
  return __builtin_riscv_cv_alu_addN(a, b, 0);
}

// CIR-LABEL: @test_alu_adduN(
// CIR: cir.call_llvm_intrinsic "riscv.cv.alu.adduN" {{%.*}}, {{%.*}}, {{%.*}} : (!u32i, !u32i, !u32i) -> !u32i
// LLVM-LABEL: @test_alu_adduN(
// LLVM: call i32 @llvm.riscv.cv.alu.adduN(i32 {{%.*}}, i32 {{%.*}}, i32 0)
unsigned int test_alu_adduN(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_alu_adduN(a, b, 0);
}

// CIR-LABEL: @test_alu_addRN(
// CIR: cir.call_llvm_intrinsic "riscv.cv.alu.addRN" {{%.*}}, {{%.*}}, {{%.*}} : (!s32i, !s32i, !u32i) -> !s32i
// LLVM-LABEL: @test_alu_addRN(
// LLVM: call i32 @llvm.riscv.cv.alu.addRN(i32 {{%.*}}, i32 {{%.*}}, i32 0)
int test_alu_addRN(int32_t a, int32_t b) {
  return __builtin_riscv_cv_alu_addRN(a, b, 0);
}

// CIR-LABEL: @test_alu_adduRN(
// CIR: cir.call_llvm_intrinsic "riscv.cv.alu.adduRN" {{%.*}}, {{%.*}}, {{%.*}} : (!u32i, !u32i, !u32i) -> !u32i
// LLVM-LABEL: @test_alu_adduRN(
// LLVM: call i32 @llvm.riscv.cv.alu.adduRN(i32 {{%.*}}, i32 {{%.*}}, i32 0)
unsigned int test_alu_adduRN(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_alu_adduRN(a, b, 0);
}

// CIR-LABEL: @test_alu_subN(
// CIR: cir.call_llvm_intrinsic "riscv.cv.alu.subN" {{%.*}}, {{%.*}}, {{%.*}} : (!s32i, !s32i, !u32i) -> !s32i
// LLVM-LABEL: @test_alu_subN(
// LLVM: call i32 @llvm.riscv.cv.alu.subN(i32 {{%.*}}, i32 {{%.*}}, i32 0)
int test_alu_subN(int32_t a, int32_t b) {
  return __builtin_riscv_cv_alu_subN(a, b, 0);
}

// CIR-LABEL: @test_alu_subuN(
// CIR: cir.call_llvm_intrinsic "riscv.cv.alu.subuN" {{%.*}}, {{%.*}}, {{%.*}} : (!u32i, !u32i, !u32i) -> !u32i
// LLVM-LABEL: @test_alu_subuN(
// LLVM: call i32 @llvm.riscv.cv.alu.subuN(i32 {{%.*}}, i32 {{%.*}}, i32 0)
unsigned int test_alu_subuN(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_alu_subuN(a, b, 0);
}

// CIR-LABEL: @test_alu_subRN(
// CIR: cir.call_llvm_intrinsic "riscv.cv.alu.subRN" {{%.*}}, {{%.*}}, {{%.*}} : (!s32i, !s32i, !u32i) -> !s32i
// LLVM-LABEL: @test_alu_subRN(
// LLVM: call i32 @llvm.riscv.cv.alu.subRN(i32 {{%.*}}, i32 {{%.*}}, i32 0)
int test_alu_subRN(int32_t a, int32_t b) {
  return __builtin_riscv_cv_alu_subRN(a, b, 0);
}

// CIR-LABEL: @test_alu_subuRN(
// CIR: cir.call_llvm_intrinsic "riscv.cv.alu.subuRN" {{%.*}}, {{%.*}}, {{%.*}} : (!u32i, !u32i, !u32i) -> !u32i
// LLVM-LABEL: @test_alu_subuRN(
// LLVM: call i32 @llvm.riscv.cv.alu.subuRN(i32 {{%.*}}, i32 {{%.*}}, i32 0)
unsigned int test_alu_subuRN(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_alu_subuRN(a, b, 0);
}
