// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H
#include <immintrin.h>

int test_bit_scan_forward(int a) {
  return _bit_scan_forward(a);
// CHECK: @test_bit_scan_forward
// CHECK: call i32 @llvm.x86.bit.scan.forward
}

int test_bit_scan_reverse(int a) {
  return _bit_scan_reverse(a);
// CHECK: @test_bit_scan_reverse
// CHECK: call i32 @llvm.x86.bit.scan.reverse
}
