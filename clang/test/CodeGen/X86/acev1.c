// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +acev1 \
// RUN: -target-feature +avx512f -emit-llvm -o - -Werror -pedantic | FileCheck %s

// Tests macro-based ACE v1 intrinsics emit correct LLVM IR

#include <immintrin.h>

void test_acev1_outer_products(void) {
  // CHECK-LABEL: @test_acev1_outer_products
  // CHECK: call void @llvm.x86.top2bf16ps(i8 0,
  // CHECK: call void @llvm.x86.top4buud(i8 1,
  // CHECK: call void @llvm.x86.top4busd(i8 2,
  // CHECK: call void @llvm.x86.top4bssd(i8 3,
  // CHECK: call void @llvm.x86.top4bsud(i8 4,
  _tile_top2bf16ps(0, 1, 2);
  _tile_top4buud(1, 2, 3);
  _tile_top4busd(2, 3, 4);
  _tile_top4bssd(3, 4, 5);
  _tile_top4bsud(4, 5, 6);
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

void test_acev1_bsr(void) {
  // CHECK-LABEL: @test_acev1_bsr
  // CHECK: call void @llvm.x86.bsrinit()
  _bsr0_init();
}
