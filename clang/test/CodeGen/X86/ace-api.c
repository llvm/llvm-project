// RUN: %clang_cc1 %s -flax-vector-conversions=none -ffreestanding -triple=x86_64-unknown-unknown \
// RUN:   -target-feature +acev1 -target-feature +avx512f -target-feature +avx512bf16 \
// RUN:   -emit-llvm -o - -Werror | FileCheck %s

// Test ACE struct-based API functions using __acetile type
// ACE v1 uses fixed 16x64 tile dimensions (Palette 2)

#include <immintrin.h>

// CHECK-LABEL: @test_ace_top4buud
// CHECK: call x86_amx @llvm.x86.cast.vector.to.tile.v256i32(<256 x i32> %{{.*}})
// CHECK: call x86_amx @llvm.x86.top4buud.internal(i16 16, i16 64, i16 64, x86_amx %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}})
// CHECK: call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx %{{.*}})
void test_ace_top4buud(__acetile *dst, __m512i a, __m512i b) {
  __tile_ace_top4buud(dst, a, b);
}

// CHECK-LABEL: @test_ace_top4busd
// CHECK: call x86_amx @llvm.x86.top4busd.internal(i16 16, i16 64, i16 64, x86_amx %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}})
void test_ace_top4busd(__acetile *dst, __m512i a, __m512i b) {
  __tile_ace_top4busd(dst, a, b);
}

// CHECK-LABEL: @test_ace_top4bssd
// CHECK: call x86_amx @llvm.x86.top4bssd.internal(i16 16, i16 64, i16 64, x86_amx %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}})
void test_ace_top4bssd(__acetile *dst, __m512i a, __m512i b) {
  __tile_ace_top4bssd(dst, a, b);
}

// CHECK-LABEL: @test_ace_top4bsud
// CHECK: call x86_amx @llvm.x86.top4bsud.internal(i16 16, i16 64, i16 64, x86_amx %{{.*}}, <64 x i8> %{{.*}}, <64 x i8> %{{.*}})
void test_ace_top4bsud(__acetile *dst, __m512i a, __m512i b) {
  __tile_ace_top4bsud(dst, a, b);
}

// CHECK-LABEL: @test_ace_top2bf16ps
// CHECK: call x86_amx @llvm.x86.top2bf16ps.internal(i16 16, i16 64, i16 64, x86_amx %{{.*}}, <32 x bfloat> %{{.*}}, <32 x bfloat> %{{.*}})
void test_ace_top2bf16ps(__acetile *dst, __m512bh a, __m512bh b) {
  __tile_ace_top2bf16ps(dst, a, b);
}

// CHECK-LABEL: @test_ace_top4mxhf8ps
// CHECK: call x86_amx @llvm.x86.top4mxhf8ps.internal(i16 16, i16 64, i16 64, i8 5, x86_amx %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
void test_ace_top4mxhf8ps(__acetile *dst, __m512i a, __m512i b) {
  __tile_ace_top4mxhf8ps(dst, a, b, 5);
}

// CHECK-LABEL: @test_ace_top4mxbhf8ps
// CHECK: call x86_amx @llvm.x86.top4mxbhf8ps.internal(i16 16, i16 64, i16 64, i8 3, x86_amx %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
void test_ace_top4mxbhf8ps(__acetile *dst, __m512i a, __m512i b) {
  __tile_ace_top4mxbhf8ps(dst, a, b, 3);
}

// CHECK-LABEL: @test_ace_top4mxhbf8ps
// CHECK: call x86_amx @llvm.x86.top4mxhbf8ps.internal(i16 16, i16 64, i16 64, i8 7, x86_amx %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
void test_ace_top4mxhbf8ps(__acetile *dst, __m512i a, __m512i b) {
  __tile_ace_top4mxhbf8ps(dst, a, b, 7);
}

// CHECK-LABEL: @test_ace_top4mxbf8ps
// CHECK: call x86_amx @llvm.x86.top4mxbf8ps.internal(i16 16, i16 64, i16 64, i8 1, x86_amx %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
void test_ace_top4mxbf8ps(__acetile *dst, __m512i a, __m512i b) {
  __tile_ace_top4mxbf8ps(dst, a, b, 1);
}

// CHECK-LABEL: @test_ace_top4mxbssps
// CHECK: call x86_amx @llvm.x86.top4mxbssps.internal(i16 16, i16 64, i16 64, i8 9, x86_amx %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
void test_ace_top4mxbssps(__acetile *dst, __m512i a, __m512i b) {
  __tile_ace_top4mxbssps(dst, a, b, 9);
}

// CHECK-LABEL: @test_ace_setcol
// CHECK: call x86_amx @llvm.x86.tilesetcol.internal(i16 16, i16 64, <16 x i32> %{{.*}}, i32 %{{.*}})
void test_ace_setcol(__acetile *dst, __m512i src) {
  __tile_ace_setcol(dst, src, 3);
}

// CHECK-LABEL: @test_ace_setrow
// CHECK: call x86_amx @llvm.x86.tilesetrow.internal(i16 16, i16 64, <16 x i32> %{{.*}}, i32 %{{.*}})
void test_ace_setrow(__acetile *dst, __m512i src) {
  __tile_ace_setrow(dst, src, 5);
}

// CHECK-LABEL: @test_ace_getrow
// CHECK: call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %{{.*}}, i32 %{{.*}})
__m512i test_ace_getrow(__acetile *src) {
  return __tile_ace_getrow(src, 7);
}

// CHECK-LABEL: @test_ace_zero
// CHECK: call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
void test_ace_zero(__acetile *dst) {
  __tile_ace_zero(dst);
}

// CHECK-LABEL: @test_ace_cvtrowd2ps
// CHECK: call <16 x float> @llvm.x86.tcvtrowd2ps.internal(i16 16, i16 64, x86_amx %{{.*}}, i32 %{{.*}})
__m512 test_ace_cvtrowd2ps(__acetile *src) {
  return __tile_ace_cvtrowd2ps(src, 2);
}

// CHECK-LABEL: @test_ace_cvtrowps2bf16h
// CHECK: call <32 x bfloat> @llvm.x86.tcvtrowps2bf16h.internal(i16 16, i16 64, x86_amx %{{.*}}, i32 %{{.*}})
__m512bh test_ace_cvtrowps2bf16h(__acetile *src) {
  return __tile_ace_cvtrowps2bf16h(src, 4);
}

// CHECK-LABEL: @test_ace_cvtrowps2bf16l
// CHECK: call <32 x bfloat> @llvm.x86.tcvtrowps2bf16l.internal(i16 16, i16 64, x86_amx %{{.*}}, i32 %{{.*}})
__m512bh test_ace_cvtrowps2bf16l(__acetile *src) {
  return __tile_ace_cvtrowps2bf16l(src, 6);
}

// CHECK-LABEL: @test_ace_cvtrowps2phh
// CHECK: call <32 x half> @llvm.x86.tcvtrowps2phh.internal(i16 16, i16 64, x86_amx %{{.*}}, i32 %{{.*}})
__m512h test_ace_cvtrowps2phh(__acetile *src) {
  return __tile_ace_cvtrowps2phh(src, 8);
}

// CHECK-LABEL: @test_ace_cvtrowps2phl
// CHECK: call <32 x half> @llvm.x86.tcvtrowps2phl.internal(i16 16, i16 64, x86_amx %{{.*}}, i32 %{{.*}})
__m512h test_ace_cvtrowps2phl(__acetile *src) {
  return __tile_ace_cvtrowps2phl(src, 10);
}

// CHECK-LABEL: @test_ace_workflow
// CHECK: call x86_amx @llvm.x86.tilezero.internal
// CHECK: call x86_amx @llvm.x86.tilesetrow.internal
// CHECK: call x86_amx @llvm.x86.top4buud.internal
// CHECK: call <16 x i32> @llvm.x86.tilemovrow.internal
void test_ace_workflow(__m512i *input, __m512i *output) {
  __acetile acc;

  // Zero the tile
  __tile_ace_zero(&acc);

  // Initialize with data via setrow
  __tile_ace_setrow(&acc, input[0], 0);

  // Perform outer product computation
  __tile_ace_top4buud(&acc, input[1], input[2]);

  // Extract result via getrow
  output[0] = __tile_ace_getrow(&acc, 0);
}

// Test BSR struct-based API functions

// CHECK-LABEL: @test_bsr_make
// CHECK: ret void
void test_bsr_make(__m512i lo, __m512i hi) {
  __bsr b = __bsr_make(lo, hi);
  (void)b;
}

// CHECK-LABEL: @test_bsr_get_lo
// CHECK: ret <8 x i64>
__m512i test_bsr_get_lo(__m512i lo, __m512i hi) {
  __bsr b = __bsr_make(lo, hi);
  return __bsr_get_lo(b);
}

// CHECK-LABEL: @test_bsr_get_hi
// CHECK: ret <8 x i64>
__m512i test_bsr_get_hi(__m512i lo, __m512i hi) {
  __bsr b = __bsr_make(lo, hi);
  return __bsr_get_hi(b);
}

// CHECK-LABEL: @test_bsr_set_lo
// CHECK: ret void
void test_bsr_set_lo(__m512i lo, __m512i hi, __m512i new_lo) {
  __bsr b = __bsr_make(lo, hi);
  b = __bsr_set_lo(b, new_lo);
  (void)b;
}

// CHECK-LABEL: @test_bsr_set_hi
// CHECK: ret void
void test_bsr_set_hi(__m512i lo, __m512i hi, __m512i new_hi) {
  __bsr b = __bsr_make(lo, hi);
  b = __bsr_set_hi(b, new_hi);
  (void)b;
}

// CHECK-LABEL: @test_bsr_store
// CHECK: call void @llvm.x86.bsrmovf(<16 x i32> %{{.*}}, <16 x i32> %{{.*}})
void test_bsr_store(__m512i lo, __m512i hi) {
  __bsr b = __bsr_make(lo, hi);
  __bsr_store(b);
}

// CHECK-LABEL: @test_bsr_load
// CHECK: call <16 x i32> @llvm.x86.bsrmovl.get()
// CHECK: call <16 x i32> @llvm.x86.bsrmovh.get()
__bsr test_bsr_load(void) {
  return __bsr_load();
}

// Test BSR-based mixed-precision outer product macros with __acetile

// CHECK-LABEL: @test_ace_bsr_top4mxhf8ps
// CHECK: call x86_bsr @llvm.x86.cast.vector.to.bsr.v32i32(<32 x i32>
// CHECK: call x86_amx @llvm.x86.top4mxhf8ps.bsr.internal(i16 16, i16 64, i16 64, i8 5, x86_amx %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}, x86_bsr %{{.*}})
void test_ace_bsr_top4mxhf8ps(__acetile *dst, __m512i a, __m512i b, __m512i lo, __m512i hi) {
  __bsr scales = __bsr_make(lo, hi);
  __tile_ace_top4mxhf8ps_bsr(dst, a, b, scales, 5);
}

// CHECK-LABEL: @test_ace_bsr_top4mxbhf8ps
// CHECK: call x86_bsr @llvm.x86.cast.vector.to.bsr.v32i32(<32 x i32>
// CHECK: call x86_amx @llvm.x86.top4mxbhf8ps.bsr.internal(i16 16, i16 64, i16 64, i8 3, x86_amx %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}, x86_bsr %{{.*}})
void test_ace_bsr_top4mxbhf8ps(__acetile *dst, __m512i a, __m512i b, __m512i lo, __m512i hi) {
  __bsr scales = __bsr_make(lo, hi);
  __tile_ace_top4mxbhf8ps_bsr(dst, a, b, scales, 3);
}

// CHECK-LABEL: @test_ace_bsr_top4mxhbf8ps
// CHECK: call x86_bsr @llvm.x86.cast.vector.to.bsr.v32i32(<32 x i32>
// CHECK: call x86_amx @llvm.x86.top4mxhbf8ps.bsr.internal(i16 16, i16 64, i16 64, i8 7, x86_amx %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}, x86_bsr %{{.*}})
void test_ace_bsr_top4mxhbf8ps(__acetile *dst, __m512i a, __m512i b, __m512i lo, __m512i hi) {
  __bsr scales = __bsr_make(lo, hi);
  __tile_ace_top4mxhbf8ps_bsr(dst, a, b, scales, 7);
}

// CHECK-LABEL: @test_ace_bsr_top4mxbf8ps
// CHECK: call x86_bsr @llvm.x86.cast.vector.to.bsr.v32i32(<32 x i32>
// CHECK: call x86_amx @llvm.x86.top4mxbf8ps.bsr.internal(i16 16, i16 64, i16 64, i8 2, x86_amx %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}, x86_bsr %{{.*}})
void test_ace_bsr_top4mxbf8ps(__acetile *dst, __m512i a, __m512i b, __m512i lo, __m512i hi) {
  __bsr scales = __bsr_make(lo, hi);
  __tile_ace_top4mxbf8ps_bsr(dst, a, b, scales, 2);
}
