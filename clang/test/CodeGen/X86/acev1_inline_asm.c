// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +acev1 \
// RUN: -target-feature +avx10.1 -emit-llvm -o - -Wall -Werror -pedantic | FileCheck %s

// Tests inline assembly with ACE v1 tile registers

#include <immintrin.h>

// TILESTORED is not available in palette 2, so the result leaves the tile one
// row at a time through TILEMOVROW and is stored from the ZMM.
void test_acev1_inline_asm_outer_product(void) {
  // CHECK-LABEL: @test_acev1_inline_asm_outer_product
  // CHECK: call void asm sideeffect "tilezero %tmm0
  // CHECK: top4buud %zmm0, %zmm1, %tmm0
  // CHECK: tilemovrow $$0, %tmm0, %zmm2
  // CHECK: vmovups %zmm2, 0(%rdi)
  __asm__ volatile (
    "tilezero %%tmm0                 \n\t"
    "top4buud %%zmm0, %%zmm1, %%tmm0 \n\t"
    "tilemovrow $0, %%tmm0, %%zmm2   \n\t"
    "vmovups %%zmm2, 0(%%rdi)        \n\t"
    ::: "memory", "tmm0", "zmm0", "zmm1", "zmm2"
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

// Tile movement is the only way in or out of an ACE tile, since ACE v1 has no
// TILELOADD/TILESTORED. Immediate index form.
void test_acev1_inline_asm_tile_movement(void) {
  // CHECK-LABEL: @test_acev1_inline_asm_tile_movement
  // CHECK: call void asm sideeffect "tilemovrow $$3, %zmm0, %tmm2
  // CHECK: tilemovcol $$5, %zmm1, %tmm3
  __asm__ volatile (
    "tilemovrow $3, %%zmm0, %%tmm2 \n\t"
    "tilemovcol $5, %%zmm1, %%tmm3 \n\t"
    ::: "memory", "tmm2", "tmm3", "zmm0", "zmm1"
  );
}

// Tile movement with the row/column index supplied in a GPR.
void test_acev1_inline_asm_tile_movement_reg_index(int idx) {
  // CHECK-LABEL: @test_acev1_inline_asm_tile_movement_reg_index
  // CHECK: call void asm sideeffect "tilemovrow $0, %zmm0, %tmm0
  // CHECK: tilemovcol $0, %zmm0, %tmm1
  __asm__ volatile (
    "tilemovrow %0, %%zmm0, %%tmm0 \n\t"
    "tilemovcol %0, %%zmm0, %%tmm1 \n\t"
    :: "r"(idx) : "memory", "tmm0", "tmm1", "zmm0"
  );
}

// BSR high/low halves, both set (ZMM to BSR) and get (BSR to ZMM) directions.
void test_acev1_inline_asm_bsr_movh_movl(void) {
  // CHECK-LABEL: @test_acev1_inline_asm_bsr_movh_movl
  // CHECK: call void asm sideeffect "bsrmovh %zmm1, %bsr0
  // CHECK: bsrmovl %zmm2, %bsr0
  // CHECK: bsrmovh %bsr0, %zmm3
  // CHECK: bsrmovl %bsr0, %zmm4
  __asm__ volatile (
    "bsrmovh %%zmm1, %%bsr0 \n\t"
    "bsrmovl %%zmm2, %%bsr0 \n\t"
    "bsrmovh %%bsr0, %%zmm3 \n\t"
    "bsrmovl %%bsr0, %%zmm4 \n\t"
    ::: "memory", "zmm1", "zmm2", "zmm3", "zmm4"
  );
}

// MX outer product, which takes the BSR scaling mode as an immediate.
void test_acev1_inline_asm_mx_outer_product(void) {
  // CHECK-LABEL: @test_acev1_inline_asm_mx_outer_product
  // CHECK: call void asm sideeffect "tilezero %tmm1
  // CHECK: top4mxhf8ps $$7, %zmm3, %zmm2, %tmm1
  __asm__ volatile (
    "tilezero %%tmm1                        \n\t"
    "top4mxhf8ps $7, %%zmm3, %%zmm2, %%tmm1 \n\t"
    ::: "memory", "tmm1", "zmm2", "zmm3"
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
