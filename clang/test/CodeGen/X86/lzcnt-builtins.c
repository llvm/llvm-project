// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +lzcnt -no-enable-noundef-analysis -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c++ -std=c++11 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +lzcnt -no-enable-noundef-analysis -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-apple-darwin -no-enable-noundef-analysis -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -x c++ -std=c++11 -ffreestanding %s -triple=x86_64-apple-darwin -no-enable-noundef-analysis -emit-llvm -o - | FileCheck %s

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

#if defined(_MSC_VER)
char ms_ctlz16_0[__lzcnt16(0x3FF0) == 2 ? 1 : -1];
char ms_ctlz16_1[__lzcnt16(0xF000) == 0 ? 1 : -1];

char ms_ctlz32_0[__lzcnt32(0xFFF00000) == 0 ? 1 : -1];
char ms_ctlz32_1[__lzcnt32(0x000001F0) == 23 ? 1 : -1];
#endif

#if defined(__LZCNT__)
char ctlz32_0[_lzcnt_u32(0x00000000) == 0 ? 1 : -1];
char ctlz32_1[_lzcnt_u32(0x000000F0) == 24 ? 1 : -1];
#endif

#ifdef __x86_64__

#if defined(_MSC_VER)
char ms_ctlz64_0[__lzcnt64(0x00000000FFF00000) == 32 ? 1 : -1];
char ms_ctlz64_1[__lzcnt64(0x00F00000000001F0) == 8 ? 1 : -1];
#endif

#if defined(__LZCNT__)
char ctlz64_0[_lzcnt_u64(0x000000000000F000ULL) == 48 ? 1 : -1];
char ctlz64_1[_lzcnt_u64(0x0100000000000001ULL) == 7 ? 1 : -1];
#endif

#endif
#endif
