// RUN: %clang_cc1 %s -flax-vector-conversions=none -ffreestanding -triple=x86_64-unknown-unknown \
// RUN: -target-feature +acev1 -target-feature +avx10.1 \
// RUN: -emit-llvm -o - -Werror -pedantic | FileCheck %s

// Tests ACE v1 tile movement operations (ZMM <-> Tile row/col).
// ACE v1 has no TILELOADD/TILESTORED, so it moves data with
// TILEMOVROW/TILEMOVCOL.

#include <immintrin.h>

// Test tile setrow - write ZMM to tile row (ACE-specific)
// Only one row changes, so the tile being written has to be read in and the
// result stored back; checking the call on its own would not notice if the
// incoming tile were dropped.
void test_tile_ace_setrow(__acetile *dst, __m512i src) {
  // A tile is 64-byte aligned, so both ends of the chain carry that alignment.
  // CHECK-LABEL: @test_tile_ace_setrow
  // CHECK: %[[OLDV:[^ ]+]] = load <256 x i32>, ptr %[[DST:[^ ]+]], align 64
  // CHECK: %[[OLD:[^ ]+]] = call x86_amx @llvm.x86.cast.vector.to.tile.v256i32(<256 x i32> %[[OLDV]])
  // CHECK: %[[NEW:[^ ]+]] = call x86_amx @llvm.x86.acev1.tilemovrowinsert.internal(i16 16, i16 64, x86_amx %[[OLD]], <16 x i32> %{{[^ ]+}}, i32 %{{[^ ]+}})
  // CHECK: %[[NEWV:[^ ]+]] = call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx %[[NEW]])
  // CHECK: store <256 x i32> %[[NEWV]], ptr %{{[^ ]+}}, align 64
  __tile_ace_setrow(dst, src, 0);
}

// Test tile setcol - write ZMM to tile column (ACE-specific)
void test_tile_ace_setcol(__acetile *dst, __m512i src) {
  // CHECK-LABEL: @test_tile_ace_setcol
  // CHECK: %[[OLDV:[^ ]+]] = load <256 x i32>, ptr %[[DST:[^ ]+]], align 64
  // CHECK: %[[OLD:[^ ]+]] = call x86_amx @llvm.x86.cast.vector.to.tile.v256i32(<256 x i32> %[[OLDV]])
  // CHECK: %[[NEW:[^ ]+]] = call x86_amx @llvm.x86.acev1.tilemovcolinsert.internal(i16 16, i16 64, x86_amx %[[OLD]], <16 x i32> %{{[^ ]+}}, i32 %{{[^ ]+}})
  // CHECK: %[[NEWV:[^ ]+]] = call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx %[[NEW]])
  // CHECK: store <256 x i32> %[[NEWV]], ptr %{{[^ ]+}}, align 64
  __tile_ace_setcol(dst, src, 0);
}

// Test tile getrow - read tile row to ZMM (ACE-specific)
__m512i test_tile_ace_getrow(__acetile *src) {
  // A read starts from memory, so the load and cast must feed the extract.
  // CHECK-LABEL: @test_tile_ace_getrow
  // CHECK: %[[OLDV:[^ ]+]] = load <256 x i32>, ptr %{{[^ ]+}}, align 64
  // CHECK: %[[OLD:[^ ]+]] = call x86_amx @llvm.x86.cast.vector.to.tile.v256i32(<256 x i32> %[[OLDV]])
  // CHECK: call <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16 16, i16 64, x86_amx %[[OLD]], i32 %{{[^ ]+}})
  return __tile_ace_getrow(src, 0);
}

// Test combined tile operations
void test_tile_movement_combined(void) {
  // Palette 2 tiles are always 16x64, so every shape here is the fixed pair.
  // CHECK-LABEL: @test_tile_movement_combined
  // CHECK: call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)
  // CHECK: call x86_amx @llvm.x86.acev1.tilemovrowinsert.internal(i16 16, i16 64,
  // CHECK: call x86_amx @llvm.x86.acev1.tilemovcolinsert.internal(i16 16, i16 64,
  __acetile tile;
  __tile_ace_zero(&tile);

  __m512i zmm = _mm512_set1_epi32(1);
  __tile_ace_setrow(&tile, zmm, 0);
  __tile_ace_setcol(&tile, zmm, 0);
}

// Integration test: complete ACE tile workflow with data movement
void test_ace_data_workflow(__m512i *input, __m512i *output) {
  // The tile goes through memory, so the cast of the row write must feed it.
  // CHECK-LABEL: @test_ace_data_workflow
  // CHECK: call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)
  // CHECK: call x86_amx @llvm.x86.acev1.tilemovrowinsert.internal(i16 16, i16 64,
  // CHECK: %[[ACCV:[^ ]+]] = load <256 x i32>, ptr %{{[^ ]+}}, align 64
  // CHECK: %[[ACC:[^ ]+]] = call x86_amx @llvm.x86.cast.vector.to.tile.v256i32(<256 x i32> %[[ACCV]])
  // CHECK: call x86_amx @llvm.x86.acev1.top4buud.internal(i16 16, i16 64, x86_amx %[[ACC]], <64 x i8> %{{[^ ]+}}, <64 x i8> %{{[^ ]+}})
  // CHECK: call <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16 16, i16 64,
  __acetile tile;
  __tile_ace_zero(&tile);

  // Initialize first row with input
  __tile_ace_setrow(&tile, input[0], 0);

  // Perform computation
  __tile_ace_top4buud(&tile, input[1], input[2]);

  // Extract result row
  output[0] = __tile_ace_getrow(&tile, 0);
}
