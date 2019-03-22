// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +popcnt -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECK-POPCNT
// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -emit-llvm -o - | FileCheck %s


#include <x86intrin.h>

#ifdef __POPCNT__
int test_mm_popcnt_u32(unsigned int __X) {
  //CHECK-POPCNT: call i32 @llvm.ctpop.i32
  return _mm_popcnt_u32(__X);
}
#endif

int test_popcnt32(unsigned int __X) {
  //CHECK: call i32 @llvm.ctpop.i32
  return _popcnt32(__X);
}

int test__popcntd(unsigned int __X) {
  //CHECK: call i32 @llvm.ctpop.i32
  return __popcntd(__X);
}

#ifdef __POPCNT__
long long test_mm_popcnt_u64(unsigned long long __X) {
  //CHECK-POPCNT: call i64 @llvm.ctpop.i64
  return _mm_popcnt_u64(__X);
}
#endif

long long test_popcnt64(unsigned long long __X) {
  //CHECK: call i64 @llvm.ctpop.i64
  return _popcnt64(__X);
}

long long test__popcntq(unsigned long long __X) {
  //CHECK: call i64 @llvm.ctpop.i64
  return __popcntq(__X);
}
