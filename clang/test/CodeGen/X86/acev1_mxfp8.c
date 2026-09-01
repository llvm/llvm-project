// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +acev1 \
// RUN: -target-feature +avx10.1 -emit-llvm -o - -Werror -pedantic | FileCheck %s

// Tests ACE v1 MX FP8 outer product macro-based intrinsics

#include <immintrin.h>

void test_top4mxhf8ps(__m512i src1, __m512i src2) {
  // CHECK-LABEL: @test_top4mxhf8ps
  // CHECK: call void @llvm.x86.top4mxhf8ps(i8 0, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}, i8 0)
  _tile_top4mxhf8ps(0, src1, src2, 0);
}

void test_top4mxbhf8ps(__m512i src1, __m512i src2) {
  // CHECK-LABEL: @test_top4mxbhf8ps
  // CHECK: call void @llvm.x86.top4mxbhf8ps(i8 1, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}, i8 1)
  _tile_top4mxbhf8ps(1, src1, src2, 1);
}

void test_top4mxhbf8ps(__m512i src1, __m512i src2) {
  // CHECK-LABEL: @test_top4mxhbf8ps
  // CHECK: call void @llvm.x86.top4mxhbf8ps(i8 2, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}, i8 0)
  _tile_top4mxhbf8ps(2, src1, src2, 0);
}

void test_top4mxbf8ps(__m512i src1, __m512i src2) {
  // CHECK-LABEL: @test_top4mxbf8ps
  // CHECK: call void @llvm.x86.top4mxbf8ps(i8 3, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}, i8 1)
  _tile_top4mxbf8ps(3, src1, src2, 1);
}

void test_top4mxbssps(__m512i src1, __m512i src2) {
  // CHECK-LABEL: @test_top4mxbssps
  // CHECK: call void @llvm.x86.top4mxbssps(i8 4, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}, i8 0)
  _tile_top4mxbssps(4, src1, src2, 0);
}

// Test all MX FP8 variants in sequence
void test_all_mxfp8_variants(__m512i src1, __m512i src2) {
  // CHECK-LABEL: @test_all_mxfp8_variants
  // CHECK: call void @llvm.x86.top4mxhf8ps
  // CHECK: call void @llvm.x86.top4mxbhf8ps
  // CHECK: call void @llvm.x86.top4mxhbf8ps
  // CHECK: call void @llvm.x86.top4mxbf8ps
  // CHECK: call void @llvm.x86.top4mxbssps
  _tile_top4mxhf8ps(0, src1, src2, 0);
  _tile_top4mxbhf8ps(3, src1, src2, 0);
  _tile_top4mxhbf8ps(6, src1, src2, 0);
  _tile_top4mxbf8ps(1, src1, src2, 1);
  _tile_top4mxbssps(4, src1, src2, 1);
}
