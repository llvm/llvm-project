// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:         -triple i686--windows -emit-llvm %s -o - \
// RUN:         | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG
// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:         -triple thumbv7--windows -emit-llvm %s -o - \
// RUN:         | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG
// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:         -triple x86_64--windows -emit-llvm %s -o - \
// RUN:         | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG
// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:         -triple i686--linux -emit-llvm %s -o - \
// RUN:         | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG
// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:         -triple x86_64--linux -emit-llvm %s -o - \
// RUN:         | FileCheck %s --check-prefixes CHECK,CHECK-64BIT-LONG
// RUN: %clang_cc1 -ffreestanding -fms-extensions \
// RUN:         -triple x86_64--darwin -emit-llvm %s -o - \
// RUN:         | FileCheck %s --check-prefixes CHECK,CHECK-64BIT-LONG

// rotate left

unsigned char test_rotl8(unsigned char value, unsigned char shift) {
  return _rotl8(value, shift);
}
// CHECK: i8 @test_rotl8
// CHECK:   [[R:%.*]] = call i8 @llvm.fshl.i8(i8 [[X:%.*]], i8 [[X]], i8 [[Y:%.*]])
// CHECK:   ret i8 [[R]]

unsigned short test_rotl16(unsigned short value, unsigned char shift) {
  return _rotl16(value, shift);
}
// CHECK: i16 @test_rotl16
// CHECK:   [[R:%.*]] = call i16 @llvm.fshl.i16(i16 [[X:%.*]], i16 [[X]], i16 [[Y:%.*]])
// CHECK:   ret i16 [[R]]

unsigned int test_rotl(unsigned int value, int shift) {
  return _rotl(value, shift);
}
// CHECK: i32 @test_rotl
// CHECK:   [[R:%.*]] = call i32 @llvm.fshl.i32(i32 [[X:%.*]], i32 [[X]], i32 [[Y:%.*]])
// CHECK:   ret i32 [[R]]

unsigned long test_lrotl(unsigned long value, int shift) {
  return _lrotl(value, shift);
}
// CHECK-32BIT-LONG: i32 @test_lrotl
// CHECK-32BIT-LONG:   [[R:%.*]] = call i32 @llvm.fshl.i32(i32 [[X:%.*]], i32 [[X]], i32 [[Y:%.*]])
// CHECK-32BIT-LONG:   ret i32 [[R]]
// CHECK-64BIT-LONG: i64 @test_lrotl
// CHECK-64BIT-LONG:   [[R:%.*]] = call i64 @llvm.fshl.i64(i64 [[X:%.*]], i64 [[X]], i64 [[Y:%.*]])
// CHECK-64BIT-LONG:   ret i64 [[R]]

unsigned __int64 test_rotl64(unsigned __int64 value, int shift) {
  return _rotl64(value, shift);
}
// CHECK: i64 @test_rotl64
// CHECK:   [[R:%.*]] = call i64 @llvm.fshl.i64(i64 [[X:%.*]], i64 [[X]], i64 [[Y:%.*]])
// CHECK:   ret i64 [[R]]

// rotate right

unsigned char test_rotr8(unsigned char value, unsigned char shift) {
  return _rotr8(value, shift);
}
// CHECK: i8 @test_rotr8
// CHECK:   [[R:%.*]] = call i8 @llvm.fshr.i8(i8 [[X:%.*]], i8 [[X]], i8 [[Y:%.*]])
// CHECK:   ret i8 [[R]]

unsigned short test_rotr16(unsigned short value, unsigned char shift) {
  return _rotr16(value, shift);
}
// CHECK: i16 @test_rotr16
// CHECK:   [[R:%.*]] = call i16 @llvm.fshr.i16(i16 [[X:%.*]], i16 [[X]], i16 [[Y:%.*]])
// CHECK:   ret i16 [[R]]

unsigned int test_rotr(unsigned int value, int shift) {
  return _rotr(value, shift);
}
// CHECK: i32 @test_rotr
// CHECK:   [[R:%.*]] = call i32 @llvm.fshr.i32(i32 [[X:%.*]], i32 [[X]], i32 [[Y:%.*]])
// CHECK:   ret i32 [[R]]

unsigned long test_lrotr(unsigned long value, int shift) {
  return _lrotr(value, shift);
}
// CHECK-32BIT-LONG: i32 @test_lrotr
// CHECK-32BIT-LONG:   [[R:%.*]] = call i32 @llvm.fshr.i32(i32 [[X:%.*]], i32 [[X]], i32 [[Y:%.*]])
// CHECK-32BIT-LONG:   ret i32 [[R]]
// CHECK-64BIT-LONG: i64 @test_lrotr
// CHECK-64BIT-LONG:   [[R:%.*]] = call i64 @llvm.fshr.i64(i64 [[X:%.*]], i64 [[X]], i64 [[Y:%.*]])
// CHECK-64BIT-LONG:   ret i64 [[R]]

unsigned __int64 test_rotr64(unsigned __int64 value, int shift) {
  return _rotr64(value, shift);
}
// CHECK: i64 @test_rotr64
// CHECK:   [[R:%.*]] = call i64 @llvm.fshr.i64(i64 [[X:%.*]], i64 [[X]], i64 [[Y:%.*]])
// CHECK:   ret i64 [[R]]

