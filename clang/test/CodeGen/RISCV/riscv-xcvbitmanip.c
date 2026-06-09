// RUN: %clang_cc1 -triple riscv32 -target-feature +xcvbitmanip -emit-llvm %s -o - \
// RUN:     -disable-O0-optnone | opt -S -passes=mem2reg | FileCheck %s

#include <stdint.h>

// CHECK-LABEL: @test_bitmanip_extract(
// CHECK: call i32 @llvm.riscv.cv.bitmanip.extract(i32 {{.*}}, i32 31)
int32_t test_bitmanip_extract(uint32_t a) {
  return __builtin_riscv_cv_bitmanip_extract(a, 31);
}

// CHECK-LABEL: @test_bitmanip_extractr(
// CHECK: call i32 @llvm.riscv.cv.bitmanip.extract(i32 {{.*}}, i32 {{.*}})
int32_t test_bitmanip_extractr(uint32_t a, uint16_t b) {
  return __builtin_riscv_cv_bitmanip_extract(a, b);
}

// CHECK-LABEL: @test_bitmanip_extractu(
// CHECK: call i32 @llvm.riscv.cv.bitmanip.extractu(i32 {{.*}}, i32 31)
uint32_t test_bitmanip_extractu(uint32_t a) {
  return __builtin_riscv_cv_bitmanip_extractu(a, 31);
}

// CHECK-LABEL: @test_bitmanip_extractur(
// CHECK: call i32 @llvm.riscv.cv.bitmanip.extractu(i32 {{.*}}, i32 {{.*}})
uint32_t test_bitmanip_extractur(uint32_t a, uint16_t b) {
  return __builtin_riscv_cv_bitmanip_extractu(a, b);
}

// CHECK-LABEL: @test_bitmanip_insert(
// CHECK: call i32 @llvm.riscv.cv.bitmanip.insert(i32 {{.*}}, i32 31, i32 {{.*}})
uint32_t test_bitmanip_insert(uint32_t a, uint32_t k) {
  return __builtin_riscv_cv_bitmanip_insert(a, 31, k);
}

// CHECK-LABEL: @test_bitmanip_insertr(
// CHECK: call i32 @llvm.riscv.cv.bitmanip.insert(i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
uint32_t test_bitmanip_insertr(uint32_t a, uint16_t b, uint32_t k) {
  return __builtin_riscv_cv_bitmanip_insert(a, b, k);
}

// CHECK-LABEL: @test_bitmanip_bclr(
// CHECK: call i32 @llvm.riscv.cv.bitmanip.bclr(i32 {{.*}}, i32 31)
uint32_t test_bitmanip_bclr(uint32_t a) {
  return __builtin_riscv_cv_bitmanip_bclr(a, 31);
}

// CHECK-LABEL: @test_bitmanip_bclrr(
// CHECK: call i32 @llvm.riscv.cv.bitmanip.bclr(i32 {{.*}}, i32 {{.*}})
uint32_t test_bitmanip_bclrr(uint32_t a, uint16_t b) {
  return __builtin_riscv_cv_bitmanip_bclr(a, b);
}

// CHECK-LABEL: @test_bitmanip_bset(
// CHECK: call i32 @llvm.riscv.cv.bitmanip.bset(i32 {{.*}}, i32 31)
uint32_t test_bitmanip_bset(uint32_t a) {
  return __builtin_riscv_cv_bitmanip_bset(a, 31);
}

// CHECK-LABEL: @test_bitmanip_bsetr(
// CHECK: call i32 @llvm.riscv.cv.bitmanip.bset(i32 {{.*}}, i32 {{.*}})
uint32_t test_bitmanip_bsetr(uint32_t a, uint16_t b) {
  return __builtin_riscv_cv_bitmanip_bset(a, b);
}

// CHECK-LABEL: @test_bitmanip_ff1(
// CHECK: call i32 @llvm.cttz.i32(i32 {{.*}}, i1 false)
uint8_t test_bitmanip_ff1(uint32_t a) {
  return __builtin_riscv_cv_bitmanip_ff1(a);
}

// CHECK-LABEL: @test_bitmanip_fl1(
// CHECK: call i32 @llvm.riscv.cv.bitmanip.fl1(i32 {{.*}})
uint8_t test_bitmanip_fl1(uint32_t a) {
  return __builtin_riscv_cv_bitmanip_fl1(a);
}

// CHECK-LABEL: @test_bitmanip_clb(
// CHECK: call i32 @llvm.riscv.cv.bitmanip.clb(i32 {{.*}})
uint8_t test_bitmanip_clb(uint32_t a) {
  return __builtin_riscv_cv_bitmanip_clb(a);
}

// CHECK-LABEL: @test_bitmanip_cnt(
// CHECK: call i32 @llvm.ctpop.i32(i32 {{.*}})
uint8_t test_bitmanip_cnt(uint32_t a) {
  return __builtin_riscv_cv_bitmanip_cnt(a);
}

// CHECK-LABEL: @test_bitmanip_ror(
// CHECK: call i32 @llvm.fshr.i32(i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
uint32_t test_bitmanip_ror(uint32_t a, uint32_t b) {
  return __builtin_riscv_cv_bitmanip_ror(a, b);
}

// CHECK-LABEL: @test_bitmanip_bitrev(
// CHECK: call i32 @llvm.riscv.cv.bitmanip.bitrev(i32 {{.*}}, i32 31, i32 3)
uint32_t test_bitmanip_bitrev(uint32_t a) {
  return __builtin_riscv_cv_bitmanip_bitrev(a, 31, 3);
}
