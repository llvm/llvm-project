// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +lzcnt -emit-llvm -o - | FileCheck %s


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
