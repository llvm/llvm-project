// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s

// PR33722
// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-unknown -fms-extensions -fms-compatibility-version=19.00 -emit-llvm -o - %s | FileCheck %s

#include <x86intrin.h>

int test_bit_scan_forward(int a) {
  return _bit_scan_forward(a);
// CHECK: @test_bit_scan_forward
// CHECK: %[[call:.*]] = call i32 @llvm.cttz.i32(i32 %{{.*}}, i1 true)
// CHECK: ret i32 %[[call]]
}

int test_bit_scan_reverse(int a) {
  return _bit_scan_reverse(a);
// CHECK:  %[[call:.*]] = call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 true)
// CHECK:  %[[sub:.*]] = sub nsw i32 31, %[[call]]
// CHECK: ret i32 %[[sub]]
}

int test__bsfd(int X) {
// CHECK: @test__bsfd
// CHECK: %[[call:.*]] = call i32 @llvm.cttz.i32(i32 %{{.*}}, i1 true)
  return __bsfd(X);
}

int test__bsfq(long long X) {
// CHECK: @test__bsfq
// CHECK: %[[call:.*]] = call i64 @llvm.cttz.i64(i64 %{{.*}}, i1 true)
  return __bsfq(X);
}

int test__bsrd(int X) {
// CHECK: @test__bsrd
// CHECK:  %[[call:.*]] = call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 true)
// CHECK:  %[[sub:.*]] = sub nsw i32 31, %[[call]]
  return __bsrd(X);
}

int test__bsrq(long long X) {
// CHECK: @test__bsrq
// CHECK:  %[[call:.*]] = call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 true)
// CHECK:  %[[cast:.*]] = trunc i64 %[[call]] to i32
// CHECK:  %[[sub:.*]] = sub nsw i32 63, %[[cast]]
  return __bsrq(X);
}
