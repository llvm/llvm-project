// RUN: %clang_cc1 %s -flax-vector-conversions=none -ffreestanding -triple=x86_64-unknown-unknown  -target-feature +amx-tile  \
// RUN: -emit-llvm -o - -Werror -pedantic | FileCheck %s --check-prefixes=CHECK

#include <immintrin.h>

char buf[1024];
#define STRIDE 32

void test_tile_loadd(short row, short col) {
  //CHECK-LABEL: @test_tile_loadd
  //CHECK-DAG: call x86_amx @llvm.x86.tileloadd64.internal
  //CHECK-DAG: call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  __tile1024i a = {row, col};
  __tile_loadd(&a, buf, STRIDE);
}

void test_tile_stream_loadd(short row, short col) {
  //CHECK-LABEL: @test_tile_stream_loadd
  //CHECK-DAG: call x86_amx @llvm.x86.tileloaddt164.internal
  //CHECK-DAG: call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  __tile1024i a = {row, col};
  __tile_stream_loadd(&a, buf, STRIDE);
}

void test_tile_stored(__tile1024i c) {
  //CHECK-LABEL: @test_tile_stored
  //CHECK-DAG: call x86_amx @llvm.x86.cast.vector.to.tile.v256i32(<256 x i32> {{%.*}})
  //CHECK-DAG: call void @llvm.x86.tilestored64.internal
  __tile_stored(buf, STRIDE, c);
}

void test_tile_zero(__tile1024i c) {
  //CHECK-LABEL: @test_tile_zero
  //CHECK-DAG: call x86_amx @llvm.x86.tilezero.internal
  //CHECK-DAG: call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  __tile_zero(&c);
}
