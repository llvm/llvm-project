// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown \
// RUN: -target-feature +amx-movrs -target-feature +avx512f \
// RUN: -emit-llvm -o - -Wall -Werror -pedantic \
// RUN: -Wno-gnu-statement-expression| FileCheck %s

#include <immintrin.h>
#include <stddef.h>

#define STRIDE 32

char buf[1024];

void test_tile_loadd(short row, short col) {
  // CHECK-LABEL: define dso_local void @test_tile_loadd(
  // CHECK:    call x86_amx @llvm.x86.tileloaddrs64.internal(i16 %{{.*}}, i16 %{{.*}}, ptr %{{.*}}, i64 %{{.*}})
  // CHECK-NEXT:    call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx %{{.*}})
  _tile_loaddrs_internal(row, col, buf, STRIDE);
}

void test_tile_loaddt1(short row, short col) {
  // CHECK-LABEL: define dso_local void @test_tile_loaddt1(
  // CHECK:    call x86_amx @llvm.x86.tileloaddrst164.internal(i16 %{{.*}}, i16 %{{.*}}, ptr %{{.*}}, i64 %{{.*}})
  // CHECK-NEXT:    call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx %{{.*}})
  _tile_loaddrst1_internal(row, col, buf, STRIDE);
}
