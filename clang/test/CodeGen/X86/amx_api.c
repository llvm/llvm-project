// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown  -target-feature +avx512f  -target-feature +amx-int8  \
// RUN: -target-feature +amx-bf16 -emit-llvm -o - -Werror -pedantic | FileCheck %s --check-prefixes=CHECK

#include <immintrin.h>

char buf[1024];
#define STRIDE 32

char buf2[1024];

// This is an example code and integration test.
void test_api(int cond, short row, short col) {
  //CHECK-LABEL: @test_api
  //CHECK: call x86_amx @llvm.x86.tileloadd64.internal
  //CHECK: call x86_amx @llvm.x86.tdpbssd.internal
  //CHECK: call void @llvm.x86.tilestored64.internal
  __tile1024i a = {row, 8};
  __tile1024i b = {8, col};
  __tile1024i c = {row, col};

  if (cond) {
    __tile_loadd(&a, buf, STRIDE);
    __tile_loadd(&b, buf, STRIDE);
    __tile_loadd(&c, buf, STRIDE);
  } else {
    __tile_loadd(&a, buf2, STRIDE);
    __tile_loadd(&b, buf2, STRIDE);
    __tile_loadd(&c, buf2, STRIDE);
  }
  __tile_dpbsud(&c, a, b);
  __tile_stored(buf, STRIDE, c);
}

void test_tile_loadd(short row, short col) {
  //CHECK-LABEL: @test_tile_loadd
  //CHECK: call x86_amx @llvm.x86.tileloadd64.internal
  //CHECK-NEXT: {{%.*}} = bitcast x86_amx {{%.*}} to <256 x i32>
  __tile1024i a = {row, col};
  __tile_loadd(&a, buf, STRIDE);
}

void test_tile_dpbsud(__tile1024i a, __tile1024i b, __tile1024i c) {
  //CHECK-LABEL: @test_tile_dpbsud
  //CHECK: call x86_amx @llvm.x86.tdpbssd.internal
  //CHECK-NEXT: {{%.*}} = bitcast x86_amx {{%.*}} to <256 x i32>
  __tile_dpbsud(&c, a, b);
}

void test_tile_stored(__tile1024i c) {
  //CHECK-LABEL: @test_tile_stored
  //CHECK: {{%.*}} = bitcast <256 x i32> {{%.*}} to x86_amx
  //CHECK-NEXT: call void @llvm.x86.tilestored64.internal
  __tile_stored(buf, STRIDE, c);
}

void test_tile_zero(__tile1024i c) {
  //CHECK-LABEL: @test_tile_zero
  //CHECK: call x86_amx @llvm.x86.tilezero.internal
  //CHECK-NEXT bitcast x86_amx {{%.*}} to <256 x i32>
  __tile_zero(&c);
}
