// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-apple-darwin -no-enable-noundef-analysis -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -x c++ -std=c++11 -ffreestanding %s -triple=x86_64-apple-darwin -no-enable-noundef-analysis -emit-llvm -o - | FileCheck %s

#include <x86intrin.h>

int test__bswapd(int X) {
// CHECK-LABEL: test__bswapd
// CHECK: call i32 @llvm.bswap.i32
  return __bswapd(X);
}

int test_bswap(int X) {
// CHECK-LABEL: test_bswap
// CHECK: call i32 @llvm.bswap.i32
  return _bswap(X);
}

long test__bswapq(long long X) {
// CHECK-LABEL: test__bswapq
// CHECK: call i64 @llvm.bswap.i64
  return __bswapq(X);
}

long test_bswap64(long long X) {
// CHECK-LABEL: test_bswap64
// CHECK: call i64 @llvm.bswap.i64
  return _bswap64(X);
}

// Test constexpr handling.
#if defined(__cplusplus) && (__cplusplus >= 201103L)

char bswapd_0[__bswapd(0x00000000) == 0x00000000 ? 1 : -1];
char bswapd_1[__bswapd(0x01020304) == 0x04030201 ? 1 : -1];

char bswap_0[_bswap(0x00000000) == 0x00000000 ? 1 : -1];
char bswap_1[_bswap(0x10203040) == 0x40302010 ? 1 : -1];

char bswapq_0[__bswapq(0x0000000000000000ULL) == 0x0000000000000000 ? 1 : -1];
char bswapq_1[__bswapq(0x0102030405060708ULL) == 0x0807060504030201 ? 1 : -1];

char bswap64_0[_bswap64(0x0000000000000000ULL) == 0x0000000000000000 ? 1 : -1];
char bswap64_1[_bswap64(0x1020304050607080ULL) == 0x8070605040302010 ? 1 : -1];

#endif
