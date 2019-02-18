// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +movbe -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECK-X64
// RUN: %clang_cc1 -ffreestanding %s -triple=i686-apple-darwin -target-feature +movbe -emit-llvm -o - | FileCheck %s


#include <immintrin.h>

short test_loadbe_i16(const short *P) {
  // CHECK-LABEL: @test_loadbe_i16
  // CHECK: [[LOAD:%.*]] = load i16, i16* %{{.*}}, align 1
  // CHECK: call i16 @llvm.bswap.i16(i16 [[LOAD]])
  return _loadbe_i16(P);
}

void test_storebe_i16(short *P, short D) {
  // CHECK-LABEL: @test_storebe_i16
  // CHECK: [[DATA:%.*]] = call i16 @llvm.bswap.i16(i16 %{{.*}})
  // CHECK: store i16 [[DATA]], i16* %{{.*}}, align 1
  _storebe_i16(P, D);
}

int test_loadbe_i32(const int *P) {
  // CHECK-LABEL: @test_loadbe_i32
  // CHECK: [[LOAD:%.*]] = load i32, i32* %{{.*}}, align 1
  // CHECK: call i32 @llvm.bswap.i32(i32 [[LOAD]])
  return _loadbe_i32(P);
}

void test_storebe_i32(int *P, int D) {
  // CHECK-LABEL: @test_storebe_i32
  // CHECK: [[DATA:%.*]] = call i32 @llvm.bswap.i32(i32 %{{.*}})
  // CHECK: store i32 [[DATA]], i32* %{{.*}}, align 1
  _storebe_i32(P, D);
}

#ifdef __x86_64__
long long test_loadbe_i64(const long long *P) {
  // CHECK-X64-LABEL: @test_loadbe_i64
  // CHECK-X64: [[LOAD:%.*]] = load i64, i64* %{{.*}}, align 1
  // CHECK-X64: call i64 @llvm.bswap.i64(i64 [[LOAD]])
  return _loadbe_i64(P);
}

void test_storebe_i64(long long *P, long long D) {
  // CHECK-X64-LABEL: @test_storebe_i64
  // CHECK-X64: [[DATA:%.*]] = call i64 @llvm.bswap.i64(i64 %{{.*}})
  // CHECK-X64: store i64 [[DATA]], i64* %{{.*}}, align 1
  _storebe_i64(P, D);
}
#endif
