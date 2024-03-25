// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +popcnt -no-enable-noundef-analysis -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECK-POPCNT
// RUN: %clang_cc1 -x c++ -std=c++11 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +popcnt -no-enable-noundef-analysis -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECK-POPCNT
// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-apple-darwin -no-enable-noundef-analysis -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -x c++ -std=c++11 -ffreestanding %s -triple=x86_64-apple-darwin -no-enable-noundef-analysis -emit-llvm -o - | FileCheck %s

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

#ifdef __x86_64__
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
#endif

// Test constexpr handling.
#if defined(__cplusplus) && (__cplusplus >= 201103L)
#if defined(__POPCNT__)
char ctpop32_0[_mm_popcnt_u32(0x00000000) == 0 ? 1 : -1];
char ctpop32_1[_mm_popcnt_u32(0x000000F0) == 4 ? 1 : -1];
#endif

char popcnt32_0[_popcnt32(0x00000000) == 0 ? 1 : -1];
char popcnt32_1[_popcnt32(0x100000F0) == 5 ? 1 : -1];

char popcntd_0[__popcntd(0x00000000) == 0 ? 1 : -1];
char popcntd_1[__popcntd(0x00F000F0) == 8 ? 1 : -1];

#ifdef __x86_64__
#if defined(__POPCNT__)
char ctpop64_0[_mm_popcnt_u64(0x0000000000000000ULL) == 0 ? 1 : -1];
char ctpop64_1[_mm_popcnt_u64(0xF000000000000001ULL) == 5 ? 1 : -1];
#endif

char popcnt64_0[_popcnt64(0x0000000000000000ULL) == 0 ? 1 : -1];
char popcnt64_1[_popcnt64(0xF00000F000000001ULL) == 9 ? 1 : -1];

char popcntq_0[__popcntq(0x0000000000000000ULL) == 0 ? 1 : -1];
char popcntq_1[__popcntq(0xF000010000300001ULL) == 8 ? 1 : -1];
#endif
#endif
