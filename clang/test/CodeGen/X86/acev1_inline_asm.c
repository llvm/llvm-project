// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +acev1 \
// RUN: -target-feature +avx10.1 -emit-llvm -o - -Wall -Werror -pedantic | FileCheck %s

// Tests ACE tile and BSR register state in inline-asm clobbers and operands.

#include <immintrin.h>

#define SRCDEST 7

// ACE has no TILESTORED; the result leaves the tile through TILEMOVROW.
void test_outer_product(void) {
  // CHECK-LABEL: @test_outer_product
  // CHECK: call void asm sideeffect "tilezero %tmm0 \0A\09
  // CHECK-SAME: top4buud %zmm0, %zmm1, %tmm0 \0A\09
  // CHECK-SAME: tilemovrow $$0, %tmm0, %zmm2 \0A\09
  // CHECK-SAME: vmovups %zmm2, 0(%rdi) \0A\09",
  // CHECK-SAME: "~{memory},~{tmm0},~{zmm0},~{zmm1},~{zmm2},~{dirflag},~{fpsr},~{flags}"()
  __asm__ volatile(
    "tilezero %%tmm0 \n\t"
    "top4buud %%zmm0, %%zmm1, %%tmm0 \n\t"
    "tilemovrow $0, %%tmm0, %%zmm2 \n\t"
    "vmovups %%zmm2, 0(%%rdi) \n\t"
    ::: "memory", "tmm0", "zmm0", "zmm1", "zmm2"
  );
}

void test_bsr_clobber(void) {
  // CHECK-LABEL: @test_bsr_clobber
  // CHECK: call void asm sideeffect "bsrinit %bsr0 \0A\09",
  // CHECK-SAME: "~{memory},~{bsr0},~{dirflag},~{fpsr},~{flags}"()
  __asm__ volatile("bsrinit %%bsr0 \n\t" ::: "memory", "bsr0");
}

void test_tile_and_bsr_clobber(void) {
  // CHECK-LABEL: @test_tile_and_bsr_clobber
  // CHECK: call void asm sideeffect "tilezero %tmm1 \0A\09
  // CHECK-SAME: top4mxhf8ps $$7, %zmm3, %zmm2, %tmm1 \0A\09",
  // CHECK-SAME: "~{memory},~{tmm1},~{bsr0},~{zmm2},~{zmm3},~{dirflag},~{fpsr},~{flags}"()
  __asm__ volatile(
    "tilezero %%tmm1 \n\t"
    "top4mxhf8ps $7, %%zmm3, %%zmm2, %%tmm1 \n\t"
    ::: "memory", "tmm1", "bsr0", "zmm2", "zmm3"
  );
}

void test_reg_index_operand(int idx) {
  // CHECK-LABEL: @test_reg_index_operand
  // CHECK: call void asm sideeffect "tilemovrow $0, %zmm0, %tmm0 \0A\09
  // CHECK-SAME: tilemovcol $0, %zmm0, %tmm1 \0A\09",
  // CHECK-SAME: "r,~{memory},~{tmm0},~{tmm1},~{zmm0},~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}})
  __asm__ volatile(
    "tilemovrow %0, %%zmm0, %%tmm0 \n\t"
    "tilemovcol %0, %%zmm0, %%tmm1 \n\t"
    :: "r"(idx) : "memory", "tmm0", "tmm1", "zmm0"
  );
}

void test_imm_tile_number(__m512 q) {
  // CHECK-LABEL: @test_imm_tile_number
  // CHECK: call void asm sideeffect "tilemovrow %eax, $0, %tmm${1:c}",
  // CHECK-SAME: "v,i,~{dirflag},~{fpsr},~{flags}"(<16 x float> %{{.*}}, i32 7)
  __asm__ volatile("tilemovrow %%eax, %0, %%tmm%c1" :: "v"(q), "i"(SRCDEST));
}

void test_tile_config(void *cfg) {
  // CHECK-LABEL: @test_tile_config
  // CHECK: call void asm sideeffect "ldtilecfg ($0) \0A\09
  // CHECK-SAME: tilezero %tmm0 \0A\09
  // CHECK-SAME: tilezero %tmm1 \0A\09
  // CHECK-SAME: tilerelease \0A\09",
  // CHECK-SAME: "r,~{memory},~{tmm0},~{tmm1},~{dirflag},~{fpsr},~{flags}"(ptr %{{.*}})
  __asm__ volatile(
    "ldtilecfg (%0) \n\t"
    "tilezero %%tmm0 \n\t"
    "tilezero %%tmm1 \n\t"
    "tilerelease \n\t"
    :: "r"(cfg) : "memory", "tmm0", "tmm1"
  );
}
