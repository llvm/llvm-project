// RUN: %clang_cc1 %s -flax-vector-conversions=none -ffreestanding -triple=x86_64-unknown-unknown \
// RUN: -target-feature +acev1 -target-feature +avx512f -target-feature +avx512bf16 \
// RUN: -emit-llvm -o - -Werror -pedantic | FileCheck %s

// Tests ACE v1 APIs using __acetile type (fixed 16x64 dimensions)
// Note: ACE v1 does not have TILELOADD/TILESTORED - uses ACE-specific tile operations

#include <immintrin.h>

// Test __acetile zero
void test_tile_ace_zero(void) {
  // CHECK-LABEL: @test_tile_ace_zero
  // CHECK: call x86_amx @llvm.x86.tilezero.internal
  // CHECK: call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  __acetile c;
  __tile_ace_zero(&c);
}

// Test TOP4BUUD ACE API
void test_tile_ace_top4buud(__acetile dst, __m512i src1, __m512i src2) {
  // CHECK-LABEL: @test_tile_ace_top4buud
  // CHECK-DAG: call x86_amx @llvm.x86.cast.vector.to.tile.v256i32(<256 x i32> {{%.*}})
  // CHECK-DAG: call x86_amx @llvm.x86.top4buud.internal
  // CHECK-DAG: call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  __tile_ace_top4buud(&dst, src1, src2);
}

// Test TOP4BUSD ACE API
void test_tile_ace_top4busd(__acetile dst, __m512i src1, __m512i src2) {
  // CHECK-LABEL: @test_tile_ace_top4busd
  // CHECK-DAG: call x86_amx @llvm.x86.cast.vector.to.tile.v256i32(<256 x i32> {{%.*}})
  // CHECK-DAG: call x86_amx @llvm.x86.top4busd.internal
  // CHECK-DAG: call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  __tile_ace_top4busd(&dst, src1, src2);
}

// Test TOP4BSSD ACE API
void test_tile_ace_top4bssd(__acetile dst, __m512i src1, __m512i src2) {
  // CHECK-LABEL: @test_tile_ace_top4bssd
  // CHECK-DAG: call x86_amx @llvm.x86.cast.vector.to.tile.v256i32(<256 x i32> {{%.*}})
  // CHECK-DAG: call x86_amx @llvm.x86.top4bssd.internal
  // CHECK-DAG: call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  __tile_ace_top4bssd(&dst, src1, src2);
}

// Test TOP4BSUD ACE API
void test_tile_ace_top4bsud(__acetile dst, __m512i src1, __m512i src2) {
  // CHECK-LABEL: @test_tile_ace_top4bsud
  // CHECK-DAG: call x86_amx @llvm.x86.cast.vector.to.tile.v256i32(<256 x i32> {{%.*}})
  // CHECK-DAG: call x86_amx @llvm.x86.top4bsud.internal
  // CHECK-DAG: call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  __tile_ace_top4bsud(&dst, src1, src2);
}

// Test TOP2BF16PS ACE API
void test_tile_ace_top2bf16ps(__acetile dst, __m512bh src1, __m512bh src2) {
  // CHECK-LABEL: @test_tile_ace_top2bf16ps
  // CHECK-DAG: call x86_amx @llvm.x86.cast.vector.to.tile.v256i32(<256 x i32> {{%.*}})
  // CHECK-DAG: call x86_amx @llvm.x86.top2bf16ps.internal
  // CHECK-DAG: call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  __tile_ace_top2bf16ps(&dst, src1, src2);
}

// Test TOP4MXHF8PS ACE API (HF8 x HF8 -> FP32 with BSR scaling)
void test_tile_ace_top4mxhf8ps(__acetile dst, __m512i src1, __m512i src2) {
  // CHECK-LABEL: @test_tile_ace_top4mxhf8ps
  // CHECK-DAG: call x86_amx @llvm.x86.cast.vector.to.tile.v256i32(<256 x i32> {{%.*}})
  // CHECK-DAG: call x86_amx @llvm.x86.top4mxhf8ps.internal(i16 16, i16 64, i16 64, i8 5
  // CHECK-DAG: call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  __tile_ace_top4mxhf8ps(&dst, src1, src2, 5);
}

// Test TOP4MXBHF8PS ACE API (BF8 x HF8 -> FP32 with BSR scaling)
void test_tile_ace_top4mxbhf8ps(__acetile dst, __m512i src1, __m512i src2) {
  // CHECK-LABEL: @test_tile_ace_top4mxbhf8ps
  // CHECK-DAG: call x86_amx @llvm.x86.cast.vector.to.tile.v256i32(<256 x i32> {{%.*}})
  // CHECK-DAG: call x86_amx @llvm.x86.top4mxbhf8ps.internal(i16 16, i16 64, i16 64, i8 3
  // CHECK-DAG: call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  __tile_ace_top4mxbhf8ps(&dst, src1, src2, 3);
}

// Test TOP4MXHBF8PS ACE API (HF8 x BF8 -> FP32 with BSR scaling)
void test_tile_ace_top4mxhbf8ps(__acetile dst, __m512i src1, __m512i src2) {
  // CHECK-LABEL: @test_tile_ace_top4mxhbf8ps
  // CHECK-DAG: call x86_amx @llvm.x86.cast.vector.to.tile.v256i32(<256 x i32> {{%.*}})
  // CHECK-DAG: call x86_amx @llvm.x86.top4mxhbf8ps.internal(i16 16, i16 64, i16 64, i8 7
  // CHECK-DAG: call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  __tile_ace_top4mxhbf8ps(&dst, src1, src2, 7);
}

// Test TOP4MXBF8PS ACE API (BF8 x BF8 -> FP32 with BSR scaling)
void test_tile_ace_top4mxbf8ps(__acetile dst, __m512i src1, __m512i src2) {
  // CHECK-LABEL: @test_tile_ace_top4mxbf8ps
  // CHECK-DAG: call x86_amx @llvm.x86.cast.vector.to.tile.v256i32(<256 x i32> {{%.*}})
  // CHECK-DAG: call x86_amx @llvm.x86.top4mxbf8ps.internal(i16 16, i16 64, i16 64, i8 1
  // CHECK-DAG: call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  __tile_ace_top4mxbf8ps(&dst, src1, src2, 1);
}

// Test TOP4MXBSSPS ACE API (MX INT8 SxS -> FP32 with BSR scaling)
void test_tile_ace_top4mxbssps(__acetile dst, __m512i src1, __m512i src2) {
  // CHECK-LABEL: @test_tile_ace_top4mxbssps
  // CHECK-DAG: call x86_amx @llvm.x86.cast.vector.to.tile.v256i32(<256 x i32> {{%.*}})
  // CHECK-DAG: call x86_amx @llvm.x86.top4mxbssps.internal(i16 16, i16 64, i16 64, i8 9
  // CHECK-DAG: call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  __tile_ace_top4mxbssps(&dst, src1, src2, 9);
}

// Test tile setrow (ACE-specific: ZMM to tile row)
void test_tile_ace_setrow(__acetile dst, __m512i src) {
  // CHECK-LABEL: @test_tile_ace_setrow
  // CHECK: call x86_amx @llvm.x86.tilesetrow.internal
  __tile_ace_setrow(&dst, src, 0);
}

// Test tile setcol (ACE-specific: ZMM to tile column)
void test_tile_ace_setcol(__acetile dst, __m512i src) {
  // CHECK-LABEL: @test_tile_ace_setcol
  // CHECK: call x86_amx @llvm.x86.tilesetcol.internal
  __tile_ace_setcol(&dst, src, 0);
}

// Test tile getrow (ACE-specific: tile row to ZMM)
__m512i test_tile_ace_getrow(__acetile src) {
  // CHECK-LABEL: @test_tile_ace_getrow
  // CHECK: call <16 x i32> @llvm.x86.tilemovrow.internal
  return __tile_ace_getrow(&src, 0);
}

// Integration test: ACE computation workflow
void test_ace_workflow(__m512i src1, __m512i src2) {
  // CHECK-LABEL: @test_ace_workflow
  // CHECK: call x86_amx @llvm.x86.tilezero.internal
  // CHECK: call x86_amx @llvm.x86.top4buud.internal
  __acetile acc;
  __tile_ace_zero(&acc);
  __tile_ace_top4buud(&acc, src1, src2);
}
