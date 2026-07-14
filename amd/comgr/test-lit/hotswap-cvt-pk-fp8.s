// COM: Test v_cvt_pk_fp8_f32 CLAMP=1 (E5M3) full conversion patch.
// COM:
// COM: Creates a minimal gfx1250 code object containing v_cvt_pk_fp8_f32
// COM: with clamp (E5M3 mode), runs the hotswap rewrite, and verifies the
// COM: replacement sequence covers: NaN detection, base F32->F16->UE5M3
// COM: conversion, RTE rounding, overflow clamping, NaN override, literal
// COM: sources, mixed literal/register sources, non-inline fractional
// COM: literals, and inline F32 constants.
// COM:
// COM: Companion tests:
// COM:   hotswap-cvt-fp8-modifiers.s - source modifier variants
// COM:   hotswap-cvt-fp8-nosled.s    - trampoline fallback path
// COM:   hotswap-cvt-fp8-multi.s     - multi-site stacking

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --dump %t.out.elf --check-idempotent 2>&1 \
// RUN:   | %FileCheck --check-prefix=API %s
// API: hotswap: liveness: kernel test_cvt_pk_fp8_literal:
// API-SAME: sgprs_before=2, sgprs_after=6
// API: REWRITE: SUCCESS
// API: IDEMPOTENT: YES

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=LOW %s
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=HIGH %s
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=NOCLAMP %s
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=LITERAL %s
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=MIXED0 %s
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=MIXED1 %s
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=INLINE %s

// ---- Kernel 1: CLAMP=1, low half (should be patched) --------------------------
//
// COM: Original site is replaced with s_branch. Trampoline body: VCC save,
// COM: two per-source F32->UE5M3 conversions (23 instructions each), pack
// COM: into 16-bit pair, merge into low half of vdst via v_bfi_b32, VCC restore.

// LOW-LABEL: <test_cvt_pk_fp8_low>:
// LOW:       s_branch
// COM: --- VCC save ---
// LOW:       s_mov_b32
// COM: --- src0 conversion ---
// LOW-NEXT:  v_and_b32{{.*}}0x7fffffff, v1
// LOW-NEXT:  v_cmp_lt_u32{{.*}}0x7f800000
// LOW-NEXT:  s_mov_b32
// LOW-NEXT:  v_max_num_f32{{.*}}, 0, v1
// LOW-NEXT:  v_cmp_le_f32{{.*}}0x47780000
// LOW-NEXT:  s_mov_b32
// LOW-NEXT:  v_mul_f32{{.*}}0x39000000
// LOW-NEXT:  v_cvt_nearest_i32_f32
// LOW-NEXT:  v_add{{.*}}0xf0
// LOW-NEXT:  v_min_u32{{.*}}0xfe
// LOW-NEXT:  v_cvt_f16_f32
// LOW-NEXT:  v_and_b32
// LOW-NEXT:  v_bfe_u32
// LOW-NEXT:  v_lshlrev_b32
// LOW-NEXT:  v_bfi_b32
// LOW-NEXT:  v_cmp_lt_u32{{.*}}0x80
// LOW-NEXT:  v_add_co_ci_u32
// LOW-NEXT:  v_min_u32{{.*}}0xfe
// LOW-NEXT:  s_mov_b32
// LOW-NEXT:  v_cndmask_b32
// LOW-NEXT:  s_mov_b32
// LOW-NEXT:  v_mov_b32
// LOW-NEXT:  v_cndmask_b32
// COM: --- src1 conversion ---
// LOW-NEXT:  v_and_b32{{.*}}0x7fffffff, v2
// LOW-NEXT:  v_cmp_lt_u32{{.*}}0x7f800000
// LOW-NEXT:  s_mov_b32
// LOW-NEXT:  v_max_num_f32{{.*}}, 0, v2
// LOW-NEXT:  v_cmp_le_f32{{.*}}0x47780000
// LOW-NEXT:  s_mov_b32
// LOW-NEXT:  v_mul_f32{{.*}}0x39000000
// LOW-NEXT:  v_cvt_nearest_i32_f32
// LOW-NEXT:  v_add{{.*}}0xf0
// LOW-NEXT:  v_min_u32{{.*}}0xfe
// LOW-NEXT:  v_cvt_f16_f32
// LOW-NEXT:  v_and_b32
// LOW-NEXT:  v_bfe_u32
// LOW-NEXT:  v_lshlrev_b32
// LOW-NEXT:  v_bfi_b32
// LOW-NEXT:  v_cmp_lt_u32{{.*}}0x80
// LOW-NEXT:  v_add_co_ci_u32
// LOW-NEXT:  v_min_u32{{.*}}0xfe
// LOW-NEXT:  s_mov_b32
// LOW-NEXT:  v_cndmask_b32
// LOW-NEXT:  s_mov_b32
// LOW-NEXT:  v_mov_b32
// LOW-NEXT:  v_cndmask_b32
// COM: --- pack + merge (low half) ---
// LOW-NEXT:  v_lshl_or_b32
// LOW-NEXT:  v_bfi_b32 v0,
// COM: --- VCC restore ---
// LOW-NEXT:  s_mov_b32

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_cvt_pk_fp8_low
.p2align 8
.type test_cvt_pk_fp8_low,@function
test_cvt_pk_fp8_low:
  v_cvt_pk_fp8_f32 v0, v1, v2 clamp
  s_endpgm
.Ltest_cvt_pk_fp8_low_end:
.size test_cvt_pk_fp8_low, .Ltest_cvt_pk_fp8_low_end-test_cvt_pk_fp8_low

// ---- Kernel 2: CLAMP=1, high half (raw encoding for op_sel[3]=1) --------------
//
// COM: Same conversion sequence as low, but final merge uses shift + bfi to
// COM: write the packed bytes into the upper 16 bits of vdst.

// HIGH-LABEL: <test_cvt_pk_fp8_high>:
// HIGH:       s_branch
// COM: --- VCC save + src0 conversion (anchor on unique src v6) ---
// HIGH:       v_and_b32{{.*}}0x7fffffff, v6
// HIGH-NEXT:  v_cmp_lt_u32{{.*}}0x7f800000
// HIGH-NEXT:  s_mov_b32
// HIGH-NEXT:  v_max_num_f32{{.*}}, 0, v6
// HIGH-NEXT:  v_cmp_le_f32{{.*}}0x47780000
// HIGH-NEXT:  s_mov_b32
// HIGH-NEXT:  v_mul_f32{{.*}}0x39000000
// HIGH-NEXT:  v_cvt_nearest_i32_f32
// HIGH-NEXT:  v_add{{.*}}0xf0
// HIGH-NEXT:  v_min_u32{{.*}}0xfe
// HIGH-NEXT:  v_cvt_f16_f32
// HIGH-NEXT:  v_and_b32
// HIGH-NEXT:  v_bfe_u32
// HIGH-NEXT:  v_lshlrev_b32
// HIGH-NEXT:  v_bfi_b32
// HIGH-NEXT:  v_cmp_lt_u32{{.*}}0x80
// HIGH-NEXT:  v_add_co_ci_u32
// HIGH-NEXT:  v_min_u32{{.*}}0xfe
// HIGH-NEXT:  s_mov_b32
// HIGH-NEXT:  v_cndmask_b32
// HIGH-NEXT:  s_mov_b32
// HIGH-NEXT:  v_mov_b32
// HIGH-NEXT:  v_cndmask_b32
// COM: --- src1 conversion ---
// HIGH-NEXT:  v_and_b32{{.*}}0x7fffffff, v7
// HIGH-NEXT:  v_cmp_lt_u32{{.*}}0x7f800000
// HIGH-NEXT:  s_mov_b32
// HIGH-NEXT:  v_max_num_f32{{.*}}, 0, v7
// HIGH-NEXT:  v_cmp_le_f32{{.*}}0x47780000
// HIGH-NEXT:  s_mov_b32
// HIGH-NEXT:  v_mul_f32{{.*}}0x39000000
// HIGH-NEXT:  v_cvt_nearest_i32_f32
// HIGH-NEXT:  v_add{{.*}}0xf0
// HIGH-NEXT:  v_min_u32{{.*}}0xfe
// HIGH-NEXT:  v_cvt_f16_f32
// HIGH-NEXT:  v_and_b32
// HIGH-NEXT:  v_bfe_u32
// HIGH-NEXT:  v_lshlrev_b32
// HIGH-NEXT:  v_bfi_b32
// HIGH-NEXT:  v_cmp_lt_u32{{.*}}0x80
// HIGH-NEXT:  v_add_co_ci_u32
// HIGH-NEXT:  v_min_u32{{.*}}0xfe
// HIGH-NEXT:  s_mov_b32
// HIGH-NEXT:  v_cndmask_b32
// HIGH-NEXT:  s_mov_b32
// HIGH-NEXT:  v_mov_b32
// HIGH-NEXT:  v_cndmask_b32
// COM: --- pack + merge (high half: shift + bfi) ---
// HIGH-NEXT:  v_lshl_or_b32
// HIGH-NEXT:  v_lshlrev_b32
// HIGH-NEXT:  v_bfi_b32 v5,
// COM: --- VCC restore ---
// HIGH-NEXT:  s_mov_b32

.globl test_cvt_pk_fp8_high
.p2align 8
.type test_cvt_pk_fp8_high,@function
test_cvt_pk_fp8_high:
  // v_cvt_pk_fp8_f32 v5, v6, v7 clamp op_sel:[0,0,0,1]
  // dword0 = 0xD769C005 (bit14=1 op_sel[3], bit15=1 CLAMP, vdst=v5)
  // dword1 = 0x02020F06 (src0=v6, src1=v7, no modifiers)
  .long 0xD769C005
  .long 0x02020F06
  s_endpgm
.Ltest_cvt_pk_fp8_high_end:
.size test_cvt_pk_fp8_high, .Ltest_cvt_pk_fp8_high_end-test_cvt_pk_fp8_high

// ---- Kernel 3: no clamp (should NOT be patched) -------------------------------

// NOCLAMP-LABEL: <test_cvt_pk_fp8_noclamp>:
// NOCLAMP-NEXT:  v_cvt_pk_fp8_f32

.globl test_cvt_pk_fp8_noclamp
.p2align 8
.type test_cvt_pk_fp8_noclamp,@function
test_cvt_pk_fp8_noclamp:
  v_cvt_pk_fp8_f32 v10, v11, v12
  s_endpgm
.Ltest_cvt_pk_fp8_noclamp_end:
.size test_cvt_pk_fp8_noclamp, .Ltest_cvt_pk_fp8_noclamp_end-test_cvt_pk_fp8_noclamp

// ---- Kernel 4: CLAMP=1 with literal F32 sources (12-byte encoding) -----------
//
// COM: Literal VOP3 operands are decoded as immediate MC operands and extend
// COM: the instruction to 12 bytes. The patch materializes the literal into
// COM: scratch VGPRs before running the normal per-source conversion sequence.
// COM: The replacement body can be emitted into any nearby NOP sled, so this
// COM: check verifies the original 12-byte slot and the generated body rather
// COM: than a specific sled label.

// LITERAL-LABEL: <test_cvt_pk_fp8_literal>:
// LITERAL-NEXT:  s_branch
// LITERAL-NEXT:  s_nop
// LITERAL-NEXT:  s_nop
// LITERAL-NEXT:  s_endpgm
// LITERAL:       v_mov_b32{{.*}}0x477f0000
// LITERAL-NEXT:  v_mov_b32{{.*}}0x477f0000
// LITERAL-NEXT:  v_and_b32{{.*}}0x7fffffff
// LITERAL:       v_cmp_le_f32{{.*}}0x47780000
// LITERAL:       v_lshl_or_b32
// LITERAL-NEXT:  v_bfi_b32 v4,
// LITERAL:       s_branch{{.*}}<test_cvt_pk_fp8_literal+0xc>

.globl test_cvt_pk_fp8_literal
.p2align 8
.type test_cvt_pk_fp8_literal,@function
test_cvt_pk_fp8_literal:
  v_cvt_pk_fp8_f32 v4, 0x477f0000, 0x477f0000 clamp
  s_endpgm
.Ltest_cvt_pk_fp8_literal_end:
.size test_cvt_pk_fp8_literal, .Ltest_cvt_pk_fp8_literal_end-test_cvt_pk_fp8_literal

// ---- Kernel 5: 12-byte literal src0, register src1 --------------------------

// MIXED0-LABEL: <test_cvt_pk_fp8_literal_src0>:
// MIXED0-NEXT:  s_branch
// MIXED0-NEXT:  s_nop
// MIXED0-NEXT:  s_nop
// MIXED0-NEXT:  s_endpgm
// MIXED0:       v_mov_b32{{.*}}0x477f0000
// MIXED0:       v_and_b32{{.*}}0x7fffffff
// MIXED0:       v_and_b32{{.*}}0x7fffffff, v17
// MIXED0:       v_bfi_b32 v16,
// MIXED0:       s_branch{{.*}}<test_cvt_pk_fp8_literal_src0+0xc>

.globl test_cvt_pk_fp8_literal_src0
.p2align 8
.type test_cvt_pk_fp8_literal_src0,@function
test_cvt_pk_fp8_literal_src0:
  v_cvt_pk_fp8_f32 v16, 0x477f0000, v17 clamp
  s_endpgm
.Ltest_cvt_pk_fp8_literal_src0_end:
.size test_cvt_pk_fp8_literal_src0, .Ltest_cvt_pk_fp8_literal_src0_end-test_cvt_pk_fp8_literal_src0

// ---- Kernel 6: register src0, 12-byte literal src1 --------------------------

// MIXED1-LABEL: <test_cvt_pk_fp8_literal_src1>:
// MIXED1-NEXT:  s_branch
// MIXED1-NEXT:  s_nop
// MIXED1-NEXT:  s_nop
// MIXED1-NEXT:  s_endpgm
// MIXED1:       v_mov_b32{{.*}}0x3eaaaaab
// MIXED1:       v_and_b32{{.*}}0x7fffffff, v21
// MIXED1:       v_and_b32{{.*}}0x7fffffff
// MIXED1:       v_bfi_b32 v20,
// MIXED1:       s_branch{{.*}}<test_cvt_pk_fp8_literal_src1+0xc>

.globl test_cvt_pk_fp8_literal_src1
.p2align 8
.type test_cvt_pk_fp8_literal_src1,@function
test_cvt_pk_fp8_literal_src1:
  v_cvt_pk_fp8_f32 v20, v21, 0.3333333432674408 clamp
  s_endpgm
.Ltest_cvt_pk_fp8_literal_src1_end:
.size test_cvt_pk_fp8_literal_src1, .Ltest_cvt_pk_fp8_literal_src1_end-test_cvt_pk_fp8_literal_src1

// ---- Kernel 7: inline fractional constants (8-byte encoding) ----------------

// INLINE-LABEL: <test_cvt_pk_fp8_inline_constants>:
// INLINE-NEXT:  s_branch
// INLINE-NEXT:  s_nop
// INLINE-NEXT:  s_endpgm
// INLINE:       v_mov_b32{{.*}}1.0
// INLINE-NEXT:  v_mov_b32{{.*}}0.5
// INLINE-NEXT:  v_and_b32{{.*}}0x7fffffff
// INLINE:       v_lshl_or_b32
// INLINE-NEXT:  v_bfi_b32 v24,
// INLINE:       s_branch{{.*}}<test_cvt_pk_fp8_inline_constants+0x8>

.globl test_cvt_pk_fp8_inline_constants
.p2align 8
.type test_cvt_pk_fp8_inline_constants,@function
test_cvt_pk_fp8_inline_constants:
  v_cvt_pk_fp8_f32 v24, 1.0, 0.5 clamp
  s_endpgm
.Ltest_cvt_pk_fp8_inline_constants_end:
.size test_cvt_pk_fp8_inline_constants, .Ltest_cvt_pk_fp8_inline_constants_end-test_cvt_pk_fp8_inline_constants

.rodata
.p2align 8
.amdhsa_kernel test_cvt_pk_fp8_low
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.amdhsa_kernel test_cvt_pk_fp8_high
  .amdhsa_next_free_vgpr 8
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.amdhsa_kernel test_cvt_pk_fp8_noclamp
  .amdhsa_next_free_vgpr 13
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.amdhsa_kernel test_cvt_pk_fp8_literal
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.amdhsa_kernel test_cvt_pk_fp8_literal_src0
  .amdhsa_next_free_vgpr 18
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.amdhsa_kernel test_cvt_pk_fp8_literal_src1
  .amdhsa_next_free_vgpr 22
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.amdhsa_kernel test_cvt_pk_fp8_inline_constants
  .amdhsa_next_free_vgpr 25
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_cvt_pk_fp8_low
      .symbol: test_cvt_pk_fp8_low.kd
      .sgpr_count: 2
      .vgpr_count: 3
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: test_cvt_pk_fp8_high
      .symbol: test_cvt_pk_fp8_high.kd
      .sgpr_count: 2
      .vgpr_count: 8
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: test_cvt_pk_fp8_noclamp
      .symbol: test_cvt_pk_fp8_noclamp.kd
      .sgpr_count: 2
      .vgpr_count: 13
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: test_cvt_pk_fp8_literal
      .symbol: test_cvt_pk_fp8_literal.kd
      .sgpr_count: 2
      .vgpr_count: 5
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: test_cvt_pk_fp8_literal_src0
      .symbol: test_cvt_pk_fp8_literal_src0.kd
      .sgpr_count: 2
      .vgpr_count: 18
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: test_cvt_pk_fp8_literal_src1
      .symbol: test_cvt_pk_fp8_literal_src1.kd
      .sgpr_count: 2
      .vgpr_count: 22
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: test_cvt_pk_fp8_inline_constants
      .symbol: test_cvt_pk_fp8_inline_constants.kd
      .sgpr_count: 2
      .vgpr_count: 25
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
