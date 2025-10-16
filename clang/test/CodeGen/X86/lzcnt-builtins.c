// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-apple-darwin -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -x c++ -std=c++11 -ffreestanding %s -triple=x86_64-apple-darwin -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-apple-darwin -emit-llvm -o - -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c++ -std=c++11 -ffreestanding %s -triple=x86_64-apple-darwin -emit-llvm -o - -fexperimental-new-constant-interpreter | FileCheck %s


#include <immintrin.h>
#include "builtin_test_helpers.h"

unsigned short test__lzcnt16(unsigned short __X)
{
  // CHECK: @llvm.ctlz.i16(i16 %{{.*}}, i1 false)
  return __lzcnt16(__X);
}
TEST_CONSTEXPR(__lzcnt16(0x0000) == 16);
TEST_CONSTEXPR(__lzcnt16(0x8000) ==  0);
TEST_CONSTEXPR(__lzcnt16(0x0010) == 11);

unsigned int test_lzcnt32(unsigned int __X)
{
  // CHECK: @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
  return __lzcnt32(__X);
}
TEST_CONSTEXPR(__lzcnt32(0x00000000) == 32);
TEST_CONSTEXPR(__lzcnt32(0x80000000) ==  0);
TEST_CONSTEXPR(__lzcnt32(0x00000010) == 27);

unsigned long long test__lzcnt64(unsigned long long __X)
{
  // CHECK: @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
  return __lzcnt64(__X);
}
TEST_CONSTEXPR(__lzcnt64(0x0000000000000000ULL) == 64);
TEST_CONSTEXPR(__lzcnt64(0x8000000000000000ULL) ==  0);
TEST_CONSTEXPR(__lzcnt64(0x0000000100000000ULL) == 31);

unsigned int test_lzcnt_u32(unsigned int __X)
{
  // CHECK: @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
  return _lzcnt_u32(__X);
}
TEST_CONSTEXPR(_lzcnt_u32(0x00000000) == 32);
TEST_CONSTEXPR(_lzcnt_u32(0x80000000) ==  0);
TEST_CONSTEXPR(_lzcnt_u32(0x00000010) == 27);

unsigned long long test__lzcnt_u64(unsigned long long __X)
{
  // CHECK: @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
  return _lzcnt_u64(__X);
}
TEST_CONSTEXPR(_lzcnt_u64(0x0000000000000000ULL) == 64);
TEST_CONSTEXPR(_lzcnt_u64(0x8000000000000000ULL) ==  0);
TEST_CONSTEXPR(_lzcnt_u64(0x0000000100000000ULL) == 31);
