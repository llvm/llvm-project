// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H
#include <immintrin.h>

int test_bit_scan_forward(int a) {
  return _bit_scan_forward(a);
// CHECK: @test_bit_scan_forward
// CHECK: %[[call:.*]] = call i32 @llvm.cttz.i32(
// CHECK: ret i32 %[[call]]
}

int test_bit_scan_reverse(int a) {
  return _bit_scan_reverse(a);
// CHECK:  %[[call:.*]] = call i32 @llvm.ctlz.i32(
// CHECK:  %[[sub:.*]] = sub nsw i32 31, %[[call]]
// CHECK: ret i32 %[[sub]]
}
