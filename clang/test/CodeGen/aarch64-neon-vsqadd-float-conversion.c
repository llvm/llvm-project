// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:  -S -disable-O0-optnone -emit-llvm -o - %s | opt -S -passes=mem2reg,dce \
// RUN: | FileCheck %s

// REQUIRES: aarch64-registered-target || arm-registered-target

#include <arm_neon.h>

// Check float conversion is accepted for int argument
uint8_t test_vsqaddb_u8(){
  return vsqaddb_u8(1, -1.0f);
}

uint16_t test_vsqaddh_u16() {
  return vsqaddh_u16(1, -1.0f);
}

uint32_t test_vsqadds_u32() {
  return vsqadds_u32(1, -1.0f);
}

uint64_t test_vsqaddd_u64() {
  return vsqaddd_u64(1, -1.0f);
}

// CHECK-LABEL: @test_vsqaddb_u8()
// CHECK: entry:
// CHECK-NEXT: [[T0:%.*]] = insertelement <8 x i8> poison, i8 1, i64 0
// CHECK-NEXT: [[T1:%.*]] = insertelement <8 x i8> poison, i8 -1, i64 0
// CHECK-NEXT: [[V:%.*]] = call <8 x i8> @llvm.aarch64.neon.usqadd.v8i8(<8 x i8> [[T0]], <8 x i8> [[T1]])
// CHECK-NEXT: [[R:%.*]] = extractelement <8 x i8> [[V]], i64 0
// CHECK-NEXT: ret i8 [[R]]

// CHECK-LABEL: @test_vsqaddh_u16()
// CHECK: entry:
// CHECK-NEXT: [[T0:%.*]] = insertelement <4 x i16> poison, i16 1, i64 0
// CHECK-NEXT: [[T1:%.*]] = insertelement <4 x i16> poison, i16 -1, i64 0
// CHECK-NEXT: [[V:%.*]]  = call <4 x i16> @llvm.aarch64.neon.usqadd.v4i16(<4 x i16> [[T0]], <4 x i16> [[T1]])
// CHECK-NEXT: [[R:%.*]] = extractelement <4 x i16> [[V]], i64 0
// CHECK-NEXT: ret i16 [[R]]

// CHECK-LABEL: @test_vsqadds_u32()
// CHECK: entry:
// CHECK-NEXT: [[V:%.*]] = call i32 @llvm.aarch64.neon.usqadd.i32(i32 1, i32 -1)
// CHECK-NEXT: ret i32 [[V]]

// CHECK-LABEL: @test_vsqaddd_u64()
// CHECK: entry:
// CHECK-NEXT: [[V:%.*]] = call i64 @llvm.aarch64.neon.usqadd.i64(i64 1, i64 -1)
// CHECK-NEXT: ret i64 [[V]]

