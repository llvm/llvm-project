// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +lzcnt -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -x c++ -std=c++11 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +lzcnt -emit-llvm -o - | FileCheck %s


#include <immintrin.h>

unsigned short test__lzcnt16(unsigned short __X)
{
  // CHECK: @llvm.ctlz.i16(i16 %{{.*}}, i1 false)
  return __lzcnt16(__X);
}

unsigned int test_lzcnt32(unsigned int __X)
{
  // CHECK: @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
  return __lzcnt32(__X);
}

unsigned long long test__lzcnt64(unsigned long long __X)
{
  // CHECK: @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
  return __lzcnt64(__X);
}

unsigned int test_lzcnt_u32(unsigned int __X)
{
  // CHECK: @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
  return _lzcnt_u32(__X);
}

unsigned long long test__lzcnt_u64(unsigned long long __X)
{
  // CHECK: @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
  return _lzcnt_u64(__X);
}


// Test constexpr handling.
#if defined(__cplusplus) && (__cplusplus >= 201103L)
char lzcnt16_0[__lzcnt16(0x0000) == 16 ? 1 : -1];
char lzcnt16_1[__lzcnt16(0x8000) ==  0 ? 1 : -1];
char lzcnt16_2[__lzcnt16(0x0010) == 11 ? 1 : -1];

char lzcnt32_0[__lzcnt32(0x00000000) == 32 ? 1 : -1];
char lzcnt32_1[__lzcnt32(0x80000000) ==  0 ? 1 : -1];
char lzcnt32_2[__lzcnt32(0x00000010) == 27 ? 1 : -1];

char lzcnt64_0[__lzcnt64(0x0000000000000000ULL) == 64 ? 1 : -1];
char lzcnt64_1[__lzcnt64(0x8000000000000000ULL) ==  0 ? 1 : -1];
char lzcnt64_2[__lzcnt64(0x0000000100000000ULL) == 31 ? 1 : -1];

char lzcntu32_0[_lzcnt_u32(0x00000000) == 32 ? 1 : -1];
char lzcntu32_1[_lzcnt_u32(0x80000000) ==  0 ? 1 : -1];
char lzcntu32_2[_lzcnt_u32(0x00000010) == 27 ? 1 : -1];

char lzcntu64_0[_lzcnt_u64(0x0000000000000000ULL) == 64 ? 1 : -1];
char lzcntu64_1[_lzcnt_u64(0x8000000000000000ULL) ==  0 ? 1 : -1];
char lzcntu64_2[_lzcnt_u64(0x0000000100000000ULL) == 31 ? 1 : -1];
#endif