// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +acev1 \
// RUN: -target-feature +avx512f -emit-llvm -o - -Wall -Werror -pedantic | FileCheck %s

// Tests inline assembly with ACE v1 tile registers

#include <immintrin.h>

void test_acev1_inline_asm_outer_product(void) {
  // CHECK-LABEL: @test_acev1_inline_asm_outer_product
  // CHECK: call void asm sideeffect "tilezero %tmm0
  // CHECK: top4buud %zmm0, %zmm1, %tmm0
  // CHECK: tilestored %tmm0
  __asm__ volatile (
    "tilezero %%tmm0               \n\t"
    "top4buud %%zmm0, %%zmm1, %%tmm0 \n\t"
    "tilestored %%tmm0, 0(%%rdi)   \n\t"
    ::: "memory", "tmm0", "zmm0", "zmm1"
  );
}

void test_acev1_inline_asm_bf16(void) {
  // CHECK-LABEL: @test_acev1_inline_asm_bf16
  // CHECK: call void asm sideeffect "tilezero %tmm1
  // CHECK: top2bf16ps %zmm2, %zmm3, %tmm1
  __asm__ volatile (
    "tilezero %%tmm1               \n\t"
    "top2bf16ps %%zmm2, %%zmm3, %%tmm1 \n\t"
    ::: "memory", "tmm1", "zmm2", "zmm3"
  );
}

void test_acev1_inline_asm_bsr(void) {
  // CHECK-LABEL: @test_acev1_inline_asm_bsr
  // CHECK: call void asm sideeffect "bsrinit %bsr0
  // CHECK: bsrmovf %zmm0, %zmm1
  __asm__ volatile (
    "bsrinit %%bsr0                \n\t"
    "bsrmovf %%zmm0, %%zmm1        \n\t"
    ::: "memory", "zmm0", "zmm1"
  );
}

void test_acev1_inline_asm_tile_config(void *cfg) {
  // CHECK-LABEL: @test_acev1_inline_asm_tile_config
  // CHECK: call void asm sideeffect "ldtilecfg ($0)
  // CHECK: tilerelease
  __asm__ volatile (
    "ldtilecfg (%0)                \n\t"
    "tilezero %%tmm0               \n\t"
    "tilezero %%tmm1               \n\t"
    "tilerelease                   \n\t"
    :: "r"(cfg) : "memory", "tmm0", "tmm1"
  );
}

void test_acev1_inline_asm_all_outer_products(void) {
  // CHECK-LABEL: @test_acev1_inline_asm_all_outer_products
  // CHECK: top4buud
  // CHECK: top4busd
  // CHECK: top4bssd
  // CHECK: top4bsud
  // CHECK: top2bf16ps
  __asm__ volatile (
    "tilezero %%tmm0               \n\t"
    "tilezero %%tmm1               \n\t"
    "tilezero %%tmm2               \n\t"
    "tilezero %%tmm3               \n\t"
    "tilezero %%tmm4               \n\t"
    "top4buud %%zmm0, %%zmm1, %%tmm0  \n\t"
    "top4busd %%zmm0, %%zmm1, %%tmm1  \n\t"
    "top4bssd %%zmm0, %%zmm1, %%tmm2  \n\t"
    "top4bsud %%zmm0, %%zmm1, %%tmm3  \n\t"
    "top2bf16ps %%zmm0, %%zmm1, %%tmm4 \n\t"
    ::: "memory", "tmm0", "tmm1", "tmm2", "tmm3", "tmm4", "zmm0", "zmm1"
  );
}
