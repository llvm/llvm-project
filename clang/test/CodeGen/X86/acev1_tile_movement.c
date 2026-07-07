// RUN: %clang_cc1 %s -flax-vector-conversions=none -ffreestanding -triple=x86_64-unknown-unknown \
// RUN: -target-feature +acev1 -target-feature +avx512f \
// RUN: -emit-llvm -o - -Werror -pedantic | FileCheck %s

// Tests ACE v1 tile movement operations (ZMM <-> Tile row/col)
// Note: ACE v1 does not have TILELOADD/TILESTORED - uses TILEMOVROW/TILEMOVCOL for data movement

#include <immintrin.h>

// Test tile setrow - write ZMM to tile row (ACE-specific)
void test_tile_ace_setrow(__acetile dst, __m512i src) {
  // CHECK-LABEL: @test_tile_ace_setrow
  // CHECK: call x86_amx @llvm.x86.tilesetrow.internal
  __tile_ace_setrow(&dst, src, 0);
}

// Test tile setrow with different row indices
void test_tile_ace_setrow_indices(__acetile dst, __m512i src) {
  // CHECK-LABEL: @test_tile_ace_setrow_indices
  // CHECK: call x86_amx @llvm.x86.tilesetrow.internal
  // CHECK: call x86_amx @llvm.x86.tilesetrow.internal
  // CHECK: call x86_amx @llvm.x86.tilesetrow.internal
  __tile_ace_setrow(&dst, src, 0);
  __tile_ace_setrow(&dst, src, 8);
  __tile_ace_setrow(&dst, src, 15);
}

// Test tile setcol - write ZMM to tile column (ACE-specific)
void test_tile_ace_setcol(__acetile dst, __m512i src) {
  // CHECK-LABEL: @test_tile_ace_setcol
  // CHECK: call x86_amx @llvm.x86.tilesetcol.internal
  __tile_ace_setcol(&dst, src, 0);
}

// Test tile setcol with different column indices
void test_tile_ace_setcol_indices(__acetile dst, __m512i src) {
  // CHECK-LABEL: @test_tile_ace_setcol_indices
  // CHECK: call x86_amx @llvm.x86.tilesetcol.internal
  // CHECK: call x86_amx @llvm.x86.tilesetcol.internal
  __tile_ace_setcol(&dst, src, 0);
  __tile_ace_setcol(&dst, src, 8);
}

// Test tile getrow - read tile row to ZMM (ACE-specific)
__m512i test_tile_ace_getrow(__acetile src) {
  // CHECK-LABEL: @test_tile_ace_getrow
  // CHECK: call <16 x i32> @llvm.x86.tilemovrow.internal
  return __tile_ace_getrow(&src, 0);
}

// Test tile getrow with different row indices
void test_tile_ace_getrow_indices(__acetile src, __m512i *out) {
  // CHECK-LABEL: @test_tile_ace_getrow_indices
  // CHECK: call <16 x i32> @llvm.x86.tilemovrow.internal
  // CHECK: call <16 x i32> @llvm.x86.tilemovrow.internal
  // CHECK: call <16 x i32> @llvm.x86.tilemovrow.internal
  out[0] = __tile_ace_getrow(&src, 0);
  out[1] = __tile_ace_getrow(&src, 8);
  out[2] = __tile_ace_getrow(&src, 15);
}

// Integration test: initialize tile using ACE tile zero and setrow
void test_tile_init_via_setrow(void) {
  // CHECK-LABEL: @test_tile_init_via_setrow
  // CHECK: call x86_amx @llvm.x86.tilezero.internal
  // CHECK: call x86_amx @llvm.x86.tilesetrow.internal
  __acetile tile;
  __tile_ace_zero(&tile);

  __m512i zmm = _mm512_set1_epi32(0x12345678);
  __tile_ace_setrow(&tile, zmm, 0);
}

// Test combined tile operations
void test_tile_movement_combined(void) {
  // CHECK-LABEL: @test_tile_movement_combined
  // CHECK: call x86_amx @llvm.x86.tilezero.internal
  // CHECK: call x86_amx @llvm.x86.tilesetrow.internal
  // CHECK: call x86_amx @llvm.x86.tilesetrow.internal
  // CHECK: call x86_amx @llvm.x86.tilesetcol.internal
  __acetile tile;
  __tile_ace_zero(&tile);

  __m512i zmm1 = _mm512_set1_epi32(1);
  __m512i zmm2 = _mm512_set1_epi32(2);

  // Set multiple rows
  __tile_ace_setrow(&tile, zmm1, 0);
  __tile_ace_setrow(&tile, zmm2, 1);

  // Set columns
  __tile_ace_setcol(&tile, zmm1, 0);
}

// Integration test: complete ACE tile workflow with data movement
void test_ace_data_workflow(__m512i *input, __m512i *output) {
  // CHECK-LABEL: @test_ace_data_workflow
  // CHECK: call x86_amx @llvm.x86.tilezero.internal
  // CHECK: call x86_amx @llvm.x86.tilesetrow.internal
  // CHECK: call x86_amx @llvm.x86.top4buud.internal
  // CHECK: call <16 x i32> @llvm.x86.tilemovrow.internal
  __acetile tile;
  __tile_ace_zero(&tile);

  // Initialize first row with input
  __tile_ace_setrow(&tile, input[0], 0);

  // Perform computation
  __tile_ace_top4buud(&tile, input[1], input[2]);

  // Extract result row
  output[0] = __tile_ace_getrow(&tile, 0);
}
