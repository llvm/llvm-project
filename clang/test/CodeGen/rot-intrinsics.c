// RUN: %clang_cc1 -ffreestanding \
// RUN:         -triple i686--linux -emit-llvm %s -o - \
// RUN:         | FileCheck %s --check-prefixes CHECK,CHECK-32BIT-LONG

// RUN: %clang_cc1 -ffreestanding \
// RUN:         -triple x86_64--linux -emit-llvm %s -o - \
// RUN:         | FileCheck %s --check-prefixes CHECK,CHECK-64BIT-LONG

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

