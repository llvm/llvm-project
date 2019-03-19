// RUN: %clang_cc1 -ffreestanding -triple i686--linux -emit-llvm %s -o - | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG
// RUN: %clang_cc1 -ffreestanding -triple x86_64--linux -emit-llvm %s -o - | FileCheck %s --check-prefixes CHECK,CHECK-64BIT-LONG
// RUN: %clang_cc1 -fms-extensions -fms-compatibility -ffreestanding %s -triple=i686-windows-msvc -target-feature +sse2 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG
// RUN: %clang_cc1 -fms-extensions -fms-compatibility -ffreestanding %s -triple=x86_64-windows-msvc -target-feature +sse2 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG
// RUN: %clang_cc1 -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 -ffreestanding %s -triple=i686-windows-msvc -target-feature +sse2 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG
// RUN: %clang_cc1 -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 -ffreestanding %s -triple=x86_64-windows-msvc -target-feature +sse2 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG

#include <x86intrin.h>

unsigned char test__rolb(unsigned char value, int shift) {
// CHECK-LABEL: i8 @test__rolb
// CHECK:   [[R:%.*]] = call i8 @llvm.fshl.i8(i8 [[X:%.*]], i8 [[X]], i8 [[Y:%.*]])
// CHECK:   ret i8 [[R]]
  return __rolb(value, shift);
}

unsigned short test__rolw(unsigned short value, int shift) {
// CHECK-LABEL: i16 @test__rolw
// CHECK:   [[R:%.*]] = call i16 @llvm.fshl.i16(i16 [[X:%.*]], i16 [[X]], i16 [[Y:%.*]])
// CHECK:   ret i16 [[R]]
  return __rolw(value, shift);
}

unsigned int test__rold(unsigned int value, int shift) {
// CHECK-LABEL: i32 @test__rold
// CHECK:   [[R:%.*]] = call i32 @llvm.fshl.i32(i32 [[X:%.*]], i32 [[X]], i32 [[Y:%.*]])
// CHECK:   ret i32 [[R]]
  return __rold(value, shift);
}

#if defined(__x86_64__)
unsigned long test__rolq(unsigned long value, int shift) {
// CHECK-LONG-LABEL: i64 @test__rolq
// CHECK-LONG:   [[R:%.*]] = call i64 @llvm.fshl.i64(i64 [[X:%.*]], i64 [[X]], i64 [[Y:%.*]])
// CHECK-LONG:   ret i64 [[R]]
  return __rolq(value, shift);
}
#endif

unsigned char test__rorb(unsigned char value, int shift) {
// CHECK-LABEL: i8 @test__rorb
// CHECK:   [[R:%.*]] = call i8 @llvm.fshr.i8(i8 [[X:%.*]], i8 [[X]], i8 [[Y:%.*]])
// CHECK:   ret i8 [[R]]
  return __rorb(value, shift);
}

unsigned short test__rorw(unsigned short value, int shift) {
// CHECK-LABEL: i16 @test__rorw
// CHECK:   [[R:%.*]] = call i16 @llvm.fshr.i16(i16 [[X:%.*]], i16 [[X]], i16 [[Y:%.*]])
// CHECK:   ret i16 [[R]]
  return __rorw(value, shift);
}

unsigned int test__rord(unsigned int value, int shift) {
// CHECK-LABEL: i32 @test__rord
// CHECK:   [[R:%.*]] = call i32 @llvm.fshr.i32(i32 [[X:%.*]], i32 [[X]], i32 [[Y:%.*]])
// CHECK:   ret i32 [[R]]
  return __rord(value, shift);
}

#if defined(__x86_64__)
unsigned long test__rorq(unsigned long value, int shift) {
// CHECK-LONG-LABEL: i64 @test__rorq
// CHECK-LONG:   [[R:%.*]] = call i64 @llvm.fshr.i64(i64 [[X:%.*]], i64 [[X]], i64 [[Y:%.*]])
// CHECK-LONG:   ret i64 [[R]]
  return __rorq(value, shift);
}
#endif

unsigned short test_rotwl(unsigned short value, int shift) {
// CHECK-LABEL: i16 @test_rotwl
// CHECK:   [[R:%.*]] = call i16 @llvm.fshl.i16(i16 [[X:%.*]], i16 [[X]], i16 [[Y:%.*]])
// CHECK:   ret i16 [[R]]
  return _rotwl(value, shift);
}

unsigned int test_rotl(unsigned int value, int shift) {
// CHECK-LABEL: i32 @test_rotl
// CHECK:   [[R:%.*]] = call i32 @llvm.fshl.i32(i32 [[X:%.*]], i32 [[X]], i32 [[Y:%.*]])
// CHECK:   ret i32 [[R]]
  return _rotl(value, shift);
}

unsigned long test_lrotl(unsigned long value, int shift) {
// CHECK-32BIT-LONG-LABEL: i32 @test_lrotl
// CHECK-32BIT-LONG:   [[R:%.*]] = call i32 @llvm.fshl.i32(i32 [[X:%.*]], i32 [[X]], i32 [[Y:%.*]])
// CHECK-32BIT-LONG:   ret i32 [[R]]
//
// CHECK-64BIT-LONG-LABEL: i64 @test_lrotl
// CHECK-64BIT-LONG:   [[R:%.*]] = call i64 @llvm.fshl.i64(i64 [[X:%.*]], i64 [[X]], i64 [[Y:%.*]])
// CHECK-64BIT-LONG:   ret i64 [[R]]
  return _lrotl(value, shift);
}


unsigned short test_rotwr(unsigned short value, int shift) {
// CHECK-LABEL: i16 @test_rotwr
// CHECK:   [[R:%.*]] = call i16 @llvm.fshr.i16(i16 [[X:%.*]], i16 [[X]], i16 [[Y:%.*]])
// CHECK:   ret i16 [[R]]
  return _rotwr(value, shift);
}

unsigned int test_rotr(unsigned int value, int shift) {
// CHECK-LABEL: i32 @test_rotr
// CHECK:   [[R:%.*]] = call i32 @llvm.fshr.i32(i32 [[X:%.*]], i32 [[X]], i32 [[Y:%.*]])
// CHECK:   ret i32 [[R]]
  return _rotr(value, shift);
}

unsigned long test_lrotr(unsigned long value, int shift) {
// CHECK-32BIT-LONG-LABEL: i32 @test_lrotr
// CHECK-32BIT-LONG:   [[R:%.*]] = call i32 @llvm.fshr.i32(i32 [[X:%.*]], i32 [[X]], i32 [[Y:%.*]])
// CHECK-32BIT-LONG:   ret i32 [[R]]
//
// CHECK-64BIT-LONG-LABEL: i64 @test_lrotr
// CHECK-64BIT-LONG:   [[R:%.*]] = call i64 @llvm.fshr.i64(i64 [[X:%.*]], i64 [[X]], i64 [[Y:%.*]])
// CHECK-64BIT-LONG:   ret i64 [[R]]
  return _lrotr(value, shift);
}

