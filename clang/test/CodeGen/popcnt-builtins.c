// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +popcnt -emit-llvm -o - | FileCheck %s


#include <immintrin.h>

unsigned int test_mm_popcnt_u32(unsigned int __X) {
  //CHECK: call i32 @llvm.ctpop.i32
  return _mm_popcnt_u32(__X);
}

unsigned int test_popcnt_32(int __X) {
  //CHECK: call i32 @llvm.ctpop.i32
  return _popcnt32(__X);
}

unsigned long long test_mm_popcnt_u64(unsigned long long __X) {
  //CHECK: call i64 @llvm.ctpop.i64
  return _mm_popcnt_u64(__X);
}

unsigned long long test_popcnt_64(long long __X) {
  //CHECK: call i64 @llvm.ctpop.i64
  return _popcnt64(__X);
}
