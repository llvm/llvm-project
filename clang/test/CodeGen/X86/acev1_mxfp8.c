// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +acev1 \
// RUN: -target-feature +avx512f -emit-llvm -o - -Werror -pedantic | FileCheck %s

// Tests ACE v1 MX FP8 outer product macro-based intrinsics

#include <immintrin.h>

void test_top4mxhf8ps(void) {
  // CHECK-LABEL: @test_top4mxhf8ps
  // CHECK: call void @llvm.x86.top4mxhf8ps(i8 0, i8 1, i8 2, i8 0)
  _tile_top4mxhf8ps(0, 1, 2, 0);
}

void test_top4mxbhf8ps(void) {
  // CHECK-LABEL: @test_top4mxbhf8ps
  // CHECK: call void @llvm.x86.top4mxbhf8ps(i8 1, i8 2, i8 3, i8 1)
  _tile_top4mxbhf8ps(1, 2, 3, 1);
}

void test_top4mxhbf8ps(void) {
  // CHECK-LABEL: @test_top4mxhbf8ps
  // CHECK: call void @llvm.x86.top4mxhbf8ps(i8 2, i8 3, i8 4, i8 0)
  _tile_top4mxhbf8ps(2, 3, 4, 0);
}

void test_top4mxbf8ps(void) {
  // CHECK-LABEL: @test_top4mxbf8ps
  // CHECK: call void @llvm.x86.top4mxbf8ps(i8 3, i8 4, i8 5, i8 1)
  _tile_top4mxbf8ps(3, 4, 5, 1);
}

void test_top4mxbssps(void) {
  // CHECK-LABEL: @test_top4mxbssps
  // CHECK: call void @llvm.x86.top4mxbssps(i8 4, i8 5, i8 6, i8 0)
  _tile_top4mxbssps(4, 5, 6, 0);
}

// Test all MX FP8 variants in sequence
void test_all_mxfp8_variants(void) {
  // CHECK-LABEL: @test_all_mxfp8_variants
  // CHECK: call void @llvm.x86.top4mxhf8ps
  // CHECK: call void @llvm.x86.top4mxbhf8ps
  // CHECK: call void @llvm.x86.top4mxhbf8ps
  // CHECK: call void @llvm.x86.top4mxbf8ps
  // CHECK: call void @llvm.x86.top4mxbssps
  _tile_top4mxhf8ps(0, 1, 2, 0);
  _tile_top4mxbhf8ps(3, 4, 5, 0);
  _tile_top4mxhbf8ps(6, 7, 0, 0);
  _tile_top4mxbf8ps(1, 2, 3, 1);
  _tile_top4mxbssps(4, 5, 6, 1);
}
