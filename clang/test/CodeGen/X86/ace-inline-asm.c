// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown \
// RUN:   -target-feature +acev1 -target-feature +avx512f \
// RUN:   -emit-llvm -o - -Wall -Werror | FileCheck %s

// Test ACE instructions in inline assembly

// BSR Management Instructions

// CHECK-LABEL: @test_bsrinit_asm
// CHECK: call void asm sideeffect "bsrinit", "~{dirflag},~{fpsr},~{flags}"()
void test_bsrinit_asm(void) {
  __asm__ volatile("bsrinit");
}

// CHECK-LABEL: @test_bsrmovf_asm
// CHECK: call void asm sideeffect "bsrmovf %zmm1, %zmm0", "~{dirflag},~{fpsr},~{flags}"()
void test_bsrmovf_asm(void) {
  __asm__ volatile("bsrmovf %%zmm1, %%zmm0" :::);
}

// CHECK-LABEL: @test_bsrmovh_asm
// CHECK: call void asm sideeffect "bsrmovh %zmm2", "~{dirflag},~{fpsr},~{flags}"()
void test_bsrmovh_asm(void) {
  __asm__ volatile("bsrmovh %%zmm2" :::);
}

// CHECK-LABEL: @test_bsrmovl_asm
// CHECK: call void asm sideeffect "bsrmovl %zmm3", "~{dirflag},~{fpsr},~{flags}"()
void test_bsrmovl_asm(void) {
  __asm__ volatile("bsrmovl %%zmm3" :::);
}

// Tile Movement Instructions

// CHECK-LABEL: @test_tilemovcol_asm
// CHECK: call void asm sideeffect "tilemovcol $$$$5, %zmm0, %tmm1", "~{tmm1},~{dirflag},~{fpsr},~{flags}"()
void test_tilemovcol_asm(void) {
  __asm__ volatile("tilemovcol $$5, %%zmm0, %%tmm1" ::: "tmm1");
}

// CHECK-LABEL: @test_tilemovrow_asm
// CHECK: call void asm sideeffect "tilemovrow $$$$3, %zmm0, %tmm2", "~{tmm2},~{dirflag},~{fpsr},~{flags}"()
void test_tilemovrow_asm(void) {
  __asm__ volatile("tilemovrow $$3, %%zmm0, %%tmm2" ::: "tmm2");
}

// Outer Product Instructions

// CHECK-LABEL: @test_top4buud_asm
// CHECK: call void asm sideeffect "top4buud %zmm1, %zmm0, %tmm0", "~{tmm0},~{dirflag},~{fpsr},~{flags}"()
void test_top4buud_asm(void) {
  __asm__ volatile("top4buud %%zmm1, %%zmm0, %%tmm0" ::: "tmm0");
}

// CHECK-LABEL: @test_top4bssd_asm
// CHECK: call void asm sideeffect "top4bssd %zmm2, %zmm1, %tmm1", "~{tmm1},~{dirflag},~{fpsr},~{flags}"()
void test_top4bssd_asm(void) {
  __asm__ volatile("top4bssd %%zmm2, %%zmm1, %%tmm1" ::: "tmm1");
}

// CHECK-LABEL: @test_top2bf16ps_asm
// CHECK: call void asm sideeffect "top2bf16ps %zmm3, %zmm2, %tmm2", "~{tmm2},~{dirflag},~{fpsr},~{flags}"()
void test_top2bf16ps_asm(void) {
  __asm__ volatile("top2bf16ps %%zmm3, %%zmm2, %%tmm2" ::: "tmm2");
}

// Mixed Precision Outer Product Instructions

// CHECK-LABEL: @test_top4mxhfbps_asm
// CHECK: call void asm sideeffect "top4mxhfbps $$$$5, %zmm1, %zmm0, %tmm3", "~{tmm3},~{dirflag},~{fpsr},~{flags}"()
void test_top4mxhfbps_asm(void) {
  __asm__ volatile("top4mxhfbps $$5, %%zmm1, %%zmm0, %%tmm3" ::: "tmm3");
}

// Full ACE operation sequence in inline assembly
// CHECK-LABEL: @test_ace_sequence_asm
// CHECK: call void asm sideeffect
void test_ace_sequence_asm(void) {
  __asm__ volatile(
    "bsrinit                           \n\t"
    "top4buud %%zmm1, %%zmm0, %%tmm0   \n\t"
    "top4bssd %%zmm3, %%zmm2, %%tmm1   \n\t"
    ::: "memory", "tmm0", "tmm1"
  );
}
