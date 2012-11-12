// RUN: %clang_cc1 %s -O3 -triple=x86_64-apple-darwin -target-feature +popcnt -emit-llvm -o - | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <x86intrin.h>

unsigned int test_mm_popcnt_u32(unsigned int __X) {
  // CHECK: @llvm.ctpop.i32
  return _mm_popcnt_u32(__X);
}

unsigned long long test_mm_popcnt_u64(unsigned long long __X) {
  // CHECK: @llvm.ctpop.i64
  return _mm_popcnt_u64(__X);
}
