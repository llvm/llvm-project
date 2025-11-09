// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-apple-darwin -no-enable-noundef-analysis -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -x c++ -std=c++11 -ffreestanding %s -triple=x86_64-apple-darwin -no-enable-noundef-analysis -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-apple-darwin -no-enable-noundef-analysis -emit-llvm -o - -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c++ -std=c++11 -ffreestanding %s -triple=x86_64-apple-darwin -no-enable-noundef-analysis -emit-llvm -o - -fexperimental-new-constant-interpreter | FileCheck %s


#include <x86intrin.h>
#include "builtin_test_helpers.h"

int test__bswapd(int X) {
// CHECK-LABEL: test__bswapd
// CHECK: call i32 @llvm.bswap.i32
  return __bswapd(X);
}
TEST_CONSTEXPR(__bswapd(0x00000000) == 0x00000000);
TEST_CONSTEXPR(__bswapd(0x01020304) == 0x04030201);

int test_bswap(int X) {
// CHECK-LABEL: test_bswap
// CHECK: call i32 @llvm.bswap.i32
  return _bswap(X);
}
TEST_CONSTEXPR(_bswap(0x00000000) == 0x00000000);
TEST_CONSTEXPR(_bswap(0x10203040) == 0x40302010);

long test__bswapq(long long X) {
// CHECK-LABEL: test__bswapq
// CHECK: call i64 @llvm.bswap.i64
  return __bswapq(X);
}
TEST_CONSTEXPR(__bswapq(0x0000000000000000ULL) == 0x0000000000000000);
TEST_CONSTEXPR(__bswapq(0x0102030405060708ULL) == 0x0807060504030201);

long test_bswap64(long long X) {
// CHECK-LABEL: test_bswap64
// CHECK: call i64 @llvm.bswap.i64
  return _bswap64(X);
}
TEST_CONSTEXPR(_bswap64(0x0000000000000000ULL) == 0x0000000000000000);
TEST_CONSTEXPR(_bswap64(0x1020304050607080ULL) == 0x8070605040302010);
