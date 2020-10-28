// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-unknown-unknown -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECKC
// RUN: %clang_cc1 -x c++ -std=c++11 -ffreestanding %s -triple=x86_64-unknown-unknown -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECKCPP


// PR33722
// RUN: %clang_cc1 -x c -ffreestanding %s -triple x86_64-unknown-unknown -fms-extensions -fms-compatibility-version=19.00 -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECKC
// RUN: %clang_cc1 -x c++ -ffreestanding %s -triple x86_64-unknown-unknown -fms-extensions -fms-compatibility-version=19.00 -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECKCPP

#include <x86intrin.h>

int test_bit_scan_forward(int a) {
  return _bit_scan_forward(a);
// CHECKC-LABEL: @test_bit_scan_forward
// CHECK: %[[call:.*]] = call i32 @llvm.cttz.i32(i32 %{{.*}}, i1 true)
// CHECK: ret i32 %[[call]]
}

int test_bit_scan_reverse(int a) {
  return _bit_scan_reverse(a);
// CHECKC-LABEL: @test_bit_scan_reverse
// CHECK:  %[[call:.*]] = call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 true)
// CHECK:  %[[sub:.*]] = sub nsw i32 31, %[[call]]
// CHECK: ret i32 %[[sub]]
}

int test__bsfd(int X) {
// CHECKC-LABEL: @test__bsfd
// CHECK: %[[call:.*]] = call i32 @llvm.cttz.i32(i32 %{{.*}}, i1 true)
  return __bsfd(X);
}

int test__bsfq(long long X) {
// CHECKC-LABEL: @test__bsfq
// CHECK: %[[call:.*]] = call i64 @llvm.cttz.i64(i64 %{{.*}}, i1 true)
  return __bsfq(X);
}

int test__bsrd(int X) {
// CHECKC-LABEL: @test__bsrd
// CHECK:  %[[call:.*]] = call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 true)
// CHECK:  %[[sub:.*]] = sub nsw i32 31, %[[call]]
  return __bsrd(X);
}

int test__bsrq(long long X) {
// CHECKC-LABEL: @test__bsrq
// CHECK:  %[[call:.*]] = call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 true)
// CHECK:  %[[cast:.*]] = trunc i64 %[[call]] to i32
// CHECK:  %[[sub:.*]] = sub nsw i32 63, %[[cast]]
  return __bsrq(X);
}

// Test constexpr handling.
#if defined(__cplusplus) && (__cplusplus >= 201103L)

char bsf_0[_bit_scan_forward(0x00000001) ==  0 ? 1 : -1];
char bsf_1[_bit_scan_forward(0x10000000) == 28 ? 1 : -1];

char bsr_0[_bit_scan_reverse(0x00000001) ==  0 ? 1 : -1];
char bsr_1[_bit_scan_reverse(0x01000000) == 24 ? 1 : -1];

char bsfd_0[__bsfd(0x00000008) ==  3 ? 1 : -1];
char bsfd_1[__bsfd(0x00010008) ==  3 ? 1 : -1];

char bsrd_0[__bsrd(0x00000010) ==  4 ? 1 : -1];
char bsrd_1[__bsrd(0x00100100) == 20 ? 1 : -1];

char bsfq_0[__bsfq(0x0000000800000000ULL) == 35 ? 1 : -1];
char bsfq_1[__bsfq(0x0004000000000000ULL) == 50 ? 1 : -1];

char bsrq_0[__bsrq(0x0000100800000000ULL) == 44 ? 1 : -1];
char bsrq_1[__bsrq(0x0004000100000000ULL) == 50 ? 1 : -1];

#endif
