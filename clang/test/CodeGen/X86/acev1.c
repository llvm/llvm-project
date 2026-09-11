// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +acev1 \
// RUN: -target-feature +avx10.1 -emit-llvm -o - -Werror -pedantic | FileCheck %s

// Tests macro-based ACE v1 intrinsics emit correct LLVM IR

#include <immintrin.h>

// The outer products take the accumulator tile by ID, but their two sources are
// ordinary ZMM values left to the register allocator.
void test_acev1_outer_products(__m512bh bf1, __m512bh bf2, __m512i i1,
                               __m512i i2) {
  // CHECK-LABEL: @test_acev1_outer_products
  // CHECK: call void @llvm.x86.acev1.top2bf16ps(i8 0, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}})
  // CHECK: call void @llvm.x86.acev1.top4buud(i8 1, <64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  // CHECK: call void @llvm.x86.acev1.top4busd(i8 2, <64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  // CHECK: call void @llvm.x86.acev1.top4bssd(i8 3, <64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  // CHECK: call void @llvm.x86.acev1.top4bsud(i8 4, <64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  _tile_op2bf16_ps(0, bf1, bf2);
  _tile_op4buud_epi32(1, i1, i2);
  _tile_op4busd_epi32(2, i1, i2);
  _tile_op4bssd_epi32(3, i1, i2);
  _tile_op4bsud_epi32(4, i1, i2);
}

// The mixed precision forms take an additional scale group selector.
void test_acev1_mx_outer_products(__m512i i1, __m512i i2) {
  // CHECK-LABEL: @test_acev1_mx_outer_products
  // CHECK: call void @llvm.x86.acev1.top4mxhf8ps(i8 0, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}, i8 0)
  // CHECK: call void @llvm.x86.acev1.top4mxbhf8ps(i8 1, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}, i8 1)
  // CHECK: call void @llvm.x86.acev1.top4mxhbf8ps(i8 2, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}, i8 0)
  // CHECK: call void @llvm.x86.acev1.top4mxbf8ps(i8 3, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}, i8 1)
  // CHECK: call void @llvm.x86.acev1.top4mxbssps(i8 4, <64 x i8> %{{.*}}, <64 x i8> %{{.*}}, i8 0)
  _tile_op4mxhf8_ps(0, i1, i2, 0);
  _tile_op4mxbhf8_ps(1, i1, i2, 1);
  _tile_op4mxhbf8_ps(2, i1, i2, 0);
  _tile_op4mxbf8_ps(3, i1, i2, 1);
  _tile_op4mxbss_ps(4, i1, i2, 0);
}

void test_acev1_tile_config(void *data) {
  // CHECK-LABEL: @test_acev1_tile_config
  // CHECK: call void @llvm.x86.ldtilecfg(ptr %{{.*}})
  // CHECK: call void @llvm.x86.sttilecfg(ptr %{{.*}})
  // CHECK: call void @llvm.x86.tilerelease()
  // CHECK: call void @llvm.x86.tilezero(i8 0)
  _tile_ace_loadconfig(data);
  _tile_ace_storeconfig(data);
  _tile_ace_release();
  _tile_ace_zero(0);
}

// ACE v1 has no TILELOADD/TILESTORED, so the macro API moves data through
// TILEMOVROW/TILEMOVCOL with an explicit tile register ID.
void test_acev1_tile_movement(__m512i src) {
  // CHECK-LABEL: @test_acev1_tile_movement
  // CHECK: call void @llvm.x86.acev1.tilemovrowinsert(i8 0, <16 x i32> %{{.*}}, i32 5)
  // CHECK: call void @llvm.x86.acev1.tilemovcolinsert(i8 1, <16 x i32> %{{.*}}, i32 3)
  _tile_insertrow(0, src, 5);
  _tile_insertcol(1, src, 3);
}

void test_acev1_bsr(void) {
  // CHECK-LABEL: @test_acev1_bsr
  // CHECK: call void @llvm.x86.acev1.bsr0init()
  _bsr0_init();
}
