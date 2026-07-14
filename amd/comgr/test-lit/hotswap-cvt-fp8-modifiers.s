// COM: Test source modifier variants (NEG, ABS) for v_cvt_pk_fp8_f32 and
// COM: v_cvt_sr_fp8_f32 with CLAMP=1 (E5M3 mode).
// COM:
// COM: ISA rules:
// COM:   v_cvt_pk_fp8_f32  — NEG and ABS on src0 and src1 (both F32)
// COM:   v_cvt_sr_fp8_f32  — NEG and ABS on src0 only (src1 is U32 noise)
// COM:   v_cvt_f32_fp8     — all modifiers disabled (NONEG/NOABS), excluded
// COM:
// COM: VOP3 modifier encoding (dword0 / dword1):
// COM:   ABS[0] (src0) = dword0 bit 8     NEG[0] (src0) = dword1 bit 29
// COM:   ABS[1] (src1) = dword0 bit 9     NEG[1] (src1) = dword1 bit 30
// COM:
// COM: Companion tests:
// COM:   hotswap-cvt-pk-fp8.s  — base pack conversion (no modifiers)
// COM:   hotswap-cvt-sr-fp8.s  — base SR conversion (no modifiers)

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --dump %t.out.elf --check-idempotent \
// RUN:   | %FileCheck --check-prefix=API %s
// API: REWRITE: SUCCESS
// API: IDEMPOTENT: YES

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=NEG0 %s
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=ABS1 %s
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=NEGABS %s
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=SR_NEG %s
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=SR_ABS %s

// ---- Kernel 1: v_cvt_pk_fp8_f32, NEG on src0 only ----------------------------
//
// COM: NEG[0]=1 on src0 (v1). The patch must forward the neg modifier to
// COM: v_max_num_f32 VOP3. NaN detection uses bare v1 (modifier-agnostic).

// NEG0-LABEL: <test_cvt_pk_fp8_neg_src0>:
// NEG0:       s_branch
// COM: --- VCC save ---
// NEG0:       s_mov_b32
// COM: --- src0 conversion (NEG applied to v1) ---
// NEG0:       v_and_b32{{.*}}0x7fffffff, v1
// NEG0-NEXT:  v_cmp_lt_u32{{.*}}0x7f800000
// NEG0:       v_cvt_f16_f32
// NEG0:       v_cndmask_b32
// COM: --- src1 conversion (v2, no modifier) ---
// NEG0:       v_and_b32{{.*}}0x7fffffff, v2
// NEG0-NEXT:  v_cmp_lt_u32{{.*}}0x7f800000
// NEG0:       v_cvt_f16_f32
// NEG0:       v_cndmask_b32
// COM: --- pack + merge ---
// NEG0:       v_lshl_or_b32
// NEG0-NEXT:  v_bfi_b32 v0,
// COM: --- VCC restore ---
// NEG0-NEXT:  s_mov_b32

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_cvt_pk_fp8_neg_src0
.p2align 8
.type test_cvt_pk_fp8_neg_src0,@function
test_cvt_pk_fp8_neg_src0:
  // v_cvt_pk_fp8_f32 v0, -v1, v2 clamp
  // dword0 = 0xD7698000 (CLAMP=1 bit15, vdst=v0)
  // dword1 = 0x22020501 (NEG[0]=1 bit29, src0=v1, src1=v2, src2=0x80)
  .long 0xD7698000
  .long 0x22020501
  s_endpgm
.Ltest_cvt_pk_fp8_neg_src0_end:
.size test_cvt_pk_fp8_neg_src0, .Ltest_cvt_pk_fp8_neg_src0_end-test_cvt_pk_fp8_neg_src0

// ---- Kernel 2: v_cvt_pk_fp8_f32, ABS on src1 only ----------------------------
//
// COM: ABS[1]=1 on src1 (v5). The patch must forward the abs modifier to the
// COM: src1 conversion path.

// ABS1-LABEL: <test_cvt_pk_fp8_abs_src1>:
// ABS1:       s_branch
// COM: --- src0 conversion (anchor on v4, no modifier) ---
// ABS1:       v_and_b32{{.*}}0x7fffffff, v4
// ABS1-NEXT:  v_cmp_lt_u32{{.*}}0x7f800000
// ABS1:       v_cvt_f16_f32
// ABS1:       v_cndmask_b32
// COM: --- src1 conversion (ABS applied to v5) ---
// ABS1:       v_and_b32{{.*}}0x7fffffff, v5
// ABS1-NEXT:  v_cmp_lt_u32{{.*}}0x7f800000
// ABS1:       v_cvt_f16_f32
// ABS1:       v_cndmask_b32
// COM: --- pack + merge ---
// ABS1:       v_lshl_or_b32
// ABS1-NEXT:  v_bfi_b32 v3,
// COM: --- VCC restore ---
// ABS1-NEXT:  s_mov_b32

.globl test_cvt_pk_fp8_abs_src1
.p2align 8
.type test_cvt_pk_fp8_abs_src1,@function
test_cvt_pk_fp8_abs_src1:
  // v_cvt_pk_fp8_f32 v3, v4, |v5| clamp
  // dword0 = 0xD7698203 (ABS[1]=1 bit9, CLAMP=1 bit15, vdst=v3)
  // dword1 = 0x02020B04 (src0=v4, src1=v5, src2=0x80, no NEG)
  .long 0xD7698203
  .long 0x02020B04
  s_endpgm
.Ltest_cvt_pk_fp8_abs_src1_end:
.size test_cvt_pk_fp8_abs_src1, .Ltest_cvt_pk_fp8_abs_src1_end-test_cvt_pk_fp8_abs_src1

// ---- Kernel 3: v_cvt_pk_fp8_f32, NEG+ABS on src0, NEG on src1 ----------------
//
// COM: Both sources have modifiers: -|v7| on src0 (NEG[0]+ABS[0]),
// COM: -v8 on src1 (NEG[1]). Tests simultaneous modifier forwarding.

// NEGABS-LABEL: <test_cvt_pk_fp8_negabs_both>:
// NEGABS:       s_branch
// COM: --- src0 conversion (NEG+ABS on v7, anchor on NaN detection) ---
// NEGABS:       v_and_b32{{.*}}0x7fffffff, v7
// NEGABS-NEXT:  v_cmp_lt_u32{{.*}}0x7f800000
// NEGABS:       v_cvt_f16_f32
// NEGABS:       v_cndmask_b32
// COM: --- src1 conversion (NEG on v8) ---
// NEGABS:       v_and_b32{{.*}}0x7fffffff, v8
// NEGABS-NEXT:  v_cmp_lt_u32{{.*}}0x7f800000
// NEGABS:       v_cvt_f16_f32
// NEGABS:       v_cndmask_b32
// COM: --- pack + merge ---
// NEGABS:       v_lshl_or_b32
// NEGABS-NEXT:  v_bfi_b32 v6,
// COM: --- VCC restore ---
// NEGABS-NEXT:  s_mov_b32

.globl test_cvt_pk_fp8_negabs_both
.p2align 8
.type test_cvt_pk_fp8_negabs_both,@function
test_cvt_pk_fp8_negabs_both:
  // v_cvt_pk_fp8_f32 v6, -|v7|, -v8 clamp
  // dword0 = 0xD7698106 (ABS[0]=1 bit8, CLAMP=1 bit15, vdst=v6)
  // dword1 = 0x62021107 (NEG[1]=1 bit30, NEG[0]=1 bit29, src0=v7, src1=v8, src2=0x80)
  .long 0xD7698106
  .long 0x62021107
  s_endpgm
.Ltest_cvt_pk_fp8_negabs_both_end:
.size test_cvt_pk_fp8_negabs_both, .Ltest_cvt_pk_fp8_negabs_both_end-test_cvt_pk_fp8_negabs_both

// ---- Kernel 4: v_cvt_sr_fp8_f32, NEG on src0 ---------------------------------
//
// COM: NEG[0]=1 on src0 (v11). Only src0 supports modifiers (src1 is U32).
// COM: The patch must forward neg to v_max_num_f32. Noise injection uses
// COM: v12 as the stochastic seed.

// SR_NEG-LABEL: <test_cvt_sr_fp8_neg_src0>:
// SR_NEG:       s_branch
// COM: --- NaN detection (anchor on v11) ---
// SR_NEG:       v_and_b32{{.*}}0x7fffffff, v11
// SR_NEG-NEXT:  v_cmp_lt_u32{{.*}}0x7f800000
// COM: --- Stochastic noise injection (direct F32 addition) ---
// SR_NEG:       v_lshrrev_b32{{.*}}, 12, v12
// COM: --- F32 -> F16 -> UE5M3 ---
// SR_NEG:       v_cvt_f16_f32
// COM: --- NaN override ---
// SR_NEG:       v_cndmask_b32
// COM: --- Byte merge ---
// SR_NEG:       v_bfi_b32 v10,
// COM: --- VCC restore ---
// SR_NEG-NEXT:  s_mov_b32

.globl test_cvt_sr_fp8_neg_src0
.p2align 8
.type test_cvt_sr_fp8_neg_src0,@function
test_cvt_sr_fp8_neg_src0:
  // v_cvt_sr_fp8_f32 v10, -v11, v12 clamp
  // dword0 = 0xD76B800A (CLAMP=1 bit15, vdst=v10)
  // dword1 = 0x2202190B (NEG[0]=1 bit29, src0=v11, src1=v12, src2=0x80)
  .long 0xD76B800A
  .long 0x2202190B
  s_endpgm
.Ltest_cvt_sr_fp8_neg_src0_end:
.size test_cvt_sr_fp8_neg_src0, .Ltest_cvt_sr_fp8_neg_src0_end-test_cvt_sr_fp8_neg_src0

// ---- Kernel 5: v_cvt_sr_fp8_f32, ABS on src0 ---------------------------------
//
// COM: ABS[0]=1 on src0 (v14). Tests abs modifier forwarding for the SR path.

// SR_ABS-LABEL: <test_cvt_sr_fp8_abs_src0>:
// SR_ABS:       s_branch
// COM: --- NaN detection (anchor on v14) ---
// SR_ABS:       v_and_b32{{.*}}0x7fffffff, v14
// SR_ABS-NEXT:  v_cmp_lt_u32{{.*}}0x7f800000
// COM: --- Stochastic noise injection (direct F32 addition) ---
// SR_ABS:       v_lshrrev_b32{{.*}}, 12, v15
// COM: --- F32 -> F16 -> UE5M3 ---
// SR_ABS:       v_cvt_f16_f32
// SR_ABS:       v_cndmask_b32
// COM: --- Byte merge ---
// SR_ABS:       v_bfi_b32 v13,
// COM: --- VCC restore ---
// SR_ABS-NEXT:  s_mov_b32

.globl test_cvt_sr_fp8_abs_src0
.p2align 8
.type test_cvt_sr_fp8_abs_src0,@function
test_cvt_sr_fp8_abs_src0:
  // v_cvt_sr_fp8_f32 v13, |v14|, v15 clamp
  // dword0 = 0xD76B810D (ABS[0]=1 bit8, CLAMP=1 bit15, vdst=v13)
  // dword1 = 0x02021F0E (src0=v14, src1=v15, src2=0x80, no NEG)
  .long 0xD76B810D
  .long 0x02021F0E
  s_endpgm
.Ltest_cvt_sr_fp8_abs_src0_end:
.size test_cvt_sr_fp8_abs_src0, .Ltest_cvt_sr_fp8_abs_src0_end-test_cvt_sr_fp8_abs_src0

.rodata
.p2align 8
.amdhsa_kernel test_cvt_pk_fp8_neg_src0
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.amdhsa_kernel test_cvt_pk_fp8_abs_src1
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.amdhsa_kernel test_cvt_pk_fp8_negabs_both
  .amdhsa_next_free_vgpr 9
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.amdhsa_kernel test_cvt_sr_fp8_neg_src0
  .amdhsa_next_free_vgpr 13
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.amdhsa_kernel test_cvt_sr_fp8_abs_src0
  .amdhsa_next_free_vgpr 16
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_cvt_pk_fp8_neg_src0
      .symbol: test_cvt_pk_fp8_neg_src0.kd
      .sgpr_count: 2
      .vgpr_count: 3
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: test_cvt_pk_fp8_abs_src1
      .symbol: test_cvt_pk_fp8_abs_src1.kd
      .sgpr_count: 2
      .vgpr_count: 6
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: test_cvt_pk_fp8_negabs_both
      .symbol: test_cvt_pk_fp8_negabs_both.kd
      .sgpr_count: 2
      .vgpr_count: 9
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: test_cvt_sr_fp8_neg_src0
      .symbol: test_cvt_sr_fp8_neg_src0.kd
      .sgpr_count: 2
      .vgpr_count: 13
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: test_cvt_sr_fp8_abs_src0
      .symbol: test_cvt_sr_fp8_abs_src0.kd
      .sgpr_count: 2
      .vgpr_count: 16
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
