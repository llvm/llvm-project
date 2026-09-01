// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +acev1 \
// RUN: -target-feature +avx10.1 -emit-llvm -o - -Werror -pedantic | FileCheck %s

// Tests macro-based ACE v1 intrinsics emit correct LLVM IR

#include <immintrin.h>

// The outer products take the accumulator tile by ID, but their two sources are
// ordinary ZMM values left to the register allocator.
void test_acev1_outer_products(__m512bh bf1, __m512bh bf2, __m512i i1,
                               __m512i i2) {
  // CHECK-LABEL: @test_acev1_outer_products
  // CHECK: call void @llvm.x86.top2bf16ps(i8 0, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}})
  // CHECK: call void @llvm.x86.top4buud(i8 1, <64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  // CHECK: call void @llvm.x86.top4busd(i8 2, <64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  // CHECK: call void @llvm.x86.top4bssd(i8 3, <64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  // CHECK: call void @llvm.x86.top4bsud(i8 4, <64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  _tile_top2bf16ps(0, bf1, bf2);
  _tile_top4buud(1, i1, i2);
  _tile_top4busd(2, i1, i2);
  _tile_top4bssd(3, i1, i2);
  _tile_top4bsud(4, i1, i2);
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
  // CHECK: call void @llvm.x86.tilemovrow.set(i8 0, <16 x i32> %{{.*}}, i32 5)
  // CHECK: call void @llvm.x86.tilemovcol.set(i8 1, <16 x i32> %{{.*}}, i32 3)
  _tile_setrow(0, src, 5);
  _tile_setcol(1, src, 3);
}

void test_acev1_bsr(void) {
  // CHECK-LABEL: @test_acev1_bsr
  // CHECK: call void @llvm.x86.bsrinit()
  _bsr0_init();
}
