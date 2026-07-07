// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-unknown-unknown -target-feature +acev1 -target-feature +avx512f -emit-llvm -o - | FileCheck %s

#include <immintrin.h>

// BSR Management Tests

// CHECK-LABEL: @test_bsrinit
// CHECK: call void @llvm.x86.bsrinit()
void test_bsrinit(void) {
  _bsr0_init();
}

// CHECK-LABEL: @test_bsrmovf
// CHECK: call void @llvm.x86.bsrmovf(<16 x i32> %{{.*}}, <16 x i32> %{{.*}})
void test_bsrmovf(__m512i a, __m512i b) {
  _bsr0_movf(a, b);
}

// CHECK-LABEL: @test_bsrmovh_set
// CHECK: call void @llvm.x86.bsrmovh.set(<16 x i32> %{{.*}})
void test_bsrmovh_set(__m512i a) {
  _bsr0_movh_set(a);
}

// CHECK-LABEL: @test_bsrmovh_get
// CHECK: call <16 x i32> @llvm.x86.bsrmovh.get()
__m512i test_bsrmovh_get(void) {
  return _bsr0_movh_get();
}

// CHECK-LABEL: @test_bsrmovl_set
// CHECK: call void @llvm.x86.bsrmovl.set(<16 x i32> %{{.*}})
void test_bsrmovl_set(__m512i a) {
  _bsr0_movl_set(a);
}

// CHECK-LABEL: @test_bsrmovl_get
// CHECK: call <16 x i32> @llvm.x86.bsrmovl.get()
__m512i test_bsrmovl_get(void) {
  return _bsr0_movl_get();
}

// Tile Movement Tests

// CHECK-LABEL: @test_tilemovcol
// CHECK: call void @llvm.x86.tilesetcol(i8 0, <16 x i32> %{{.*}}, i32 5)
void test_tilemovcol(__m512i src) {
  _tile_setcol(0, src, 5);
}

// CHECK-LABEL: @test_tilemovrow_to_tile
// CHECK: call void @llvm.x86.tilesetrow(i8 0, <16 x i32> %{{.*}}, i32 5)
void test_tilemovrow_to_tile(__m512i src) {
  _tile_setrow(0, src, 5);
}

// Outer Product Tests

// CHECK-LABEL: @test_top2bf16ps
// CHECK: call void @llvm.x86.top2bf16ps(i8 0, i8 1, i8 2)
void test_top2bf16ps(void) {
  _tile_top2bf16ps(0, 1, 2);
}

// CHECK-LABEL: @test_top4buud
// CHECK: call void @llvm.x86.top4buud(i8 0, i8 1, i8 2)
void test_top4buud(void) {
  _tile_top4buud(0, 1, 2);
}

// CHECK-LABEL: @test_top4busd
// CHECK: call void @llvm.x86.top4busd(i8 0, i8 1, i8 2)
void test_top4busd(void) {
  _tile_top4busd(0, 1, 2);
}

// CHECK-LABEL: @test_top4bssd
// CHECK: call void @llvm.x86.top4bssd(i8 0, i8 1, i8 2)
void test_top4bssd(void) {
  _tile_top4bssd(0, 1, 2);
}

// CHECK-LABEL: @test_top4bsud
// CHECK: call void @llvm.x86.top4bsud(i8 0, i8 1, i8 2)
void test_top4bsud(void) {
  _tile_top4bsud(0, 1, 2);
}

// Mixed Precision Outer Product Tests

// CHECK-LABEL: @test_top4mxhf8ps
// CHECK: call void @llvm.x86.top4mxhf8ps(i8 0, i8 1, i8 2, i8 5)
void test_top4mxhf8ps(void) {
  _tile_top4mxhf8ps(0, 1, 2, 5);
}

// CHECK-LABEL: @test_top4mxbhf8ps
// CHECK: call void @llvm.x86.top4mxbhf8ps(i8 0, i8 1, i8 2, i8 5)
void test_top4mxbhf8ps(void) {
  _tile_top4mxbhf8ps(0, 1, 2, 5);
}

// CHECK-LABEL: @test_top4mxhbf8ps
// CHECK: call void @llvm.x86.top4mxhbf8ps(i8 0, i8 1, i8 2, i8 5)
void test_top4mxhbf8ps(void) {
  _tile_top4mxhbf8ps(0, 1, 2, 5);
}

// CHECK-LABEL: @test_top4mxbf8ps
// CHECK: call void @llvm.x86.top4mxbf8ps(i8 0, i8 1, i8 2, i8 5)
void test_top4mxbf8ps(void) {
  _tile_top4mxbf8ps(0, 1, 2, 5);
}

// CHECK-LABEL: @test_top4mxbssps
// CHECK: call void @llvm.x86.top4mxbssps(i8 0, i8 1, i8 2, i8 5)
void test_top4mxbssps(void) {
  _tile_top4mxbssps(0, 1, 2, 5);
}
