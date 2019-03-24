// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -emit-llvm -o - | FileCheck %s

#include <x86intrin.h>

int test__bswapd(int X) {
// CHECK-LABEL: @test__bswapd
// CHECK: call i32 @llvm.bswap.i32
  return __bswapd(X);
}

int test_bswap(int X) {
// CHECK-LABEL: @test_bswap
// CHECK: call i32 @llvm.bswap.i32
  return _bswap(X);
}

long test__bswapq(long long X) {
// CHECK-LABEL: @test__bswapq
// CHECK: call i64 @llvm.bswap.i64
  return __bswapq(X);
}

long test_bswap64(long long X) {
// CHECK-LABEL: @test_bswap64
// CHECK: call i64 @llvm.bswap.i64
  return _bswap64(X);
}


