// COM: Test FP8 E5M3 patches with no NOP sled (tight instruction packing).
// COM:
// COM: Each kernel packs a CLAMP=1 FP8 conversion instruction immediately
// COM: followed by a non-NOP filler (v_mov_b32) and s_endpgm, with no NOP
// COM: padding available.  The hotswap rewriter replaces the original
// COM: instruction with s_branch to a trampoline appended after the .text
// COM: section.
// COM:
// COM: Companion tests:
// COM:   hotswap-cvt-pk-fp8.s   — v_cvt_pk_fp8_f32  base (NOP sled path)
// COM:   hotswap-cvt-sr-fp8.s   — v_cvt_sr_fp8_f32  base (NOP sled path)
// COM:   hotswap-cvt-f32-fp8.s  — v_cvt_f32_fp8     base (NOP sled path)

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --dump %t.out.elf --check-idempotent \
// RUN:   | %FileCheck --check-prefix=API %s
// API: REWRITE: SUCCESS
// API: IDEMPOTENT: YES

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=PK %s
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=SR %s
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=UNPACK %s

// ---- Kernel 1: v_cvt_pk_fp8_f32 CLAMP=1, no NOP sled -------------------------
//
// COM: No NOP padding available — trampoline is appended after .text.
// COM: Filler instruction (v_mov_b32 0x42) must be preserved between the
// COM: s_branch and s_endpgm. Branch-back lands after the original site.

// PK-LABEL: <test_nosled_pk>:
// COM: --- Original site: branch replaces v_cvt_pk_fp8_f32 ---
// PK:       s_branch
// COM: --- Filler instruction preserved ---
// PK:       v_mov_b32{{.*}}0x42
// PK:       s_endpgm
// COM: --- Trampoline: VCC save ---
// PK:       s_mov_b32
// COM: --- src0 conversion (anchor on v1) ---
// PK-NEXT:  v_and_b32{{.*}}0x7fffffff, v1
// PK-NEXT:  v_cmp_lt_u32{{.*}}0x7f800000
// COM: --- src1 conversion (anchor on v2) ---
// PK:       v_and_b32{{.*}}0x7fffffff, v2
// PK-NEXT:  v_cmp_lt_u32{{.*}}0x7f800000
// COM: --- pack + merge (low half) ---
// PK:       v_lshl_or_b32
// PK-NEXT:  v_bfi_b32 v0,
// COM: --- VCC restore + branch back ---
// PK-NEXT:  s_mov_b32
// PK-NEXT:  s_branch

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_nosled_pk
.p2align 8
.type test_nosled_pk,@function
test_nosled_pk:
  v_cvt_pk_fp8_f32 v0, v1, v2 clamp
  v_mov_b32 v3, 0x42
  s_endpgm
.Ltest_nosled_pk_end:
.size test_nosled_pk, .Ltest_nosled_pk_end-test_nosled_pk

// ---- Kernel 2: v_cvt_sr_fp8_f32 CLAMP=1, no NOP sled -------------------------
//
// COM: Stochastic-round patch in trampoline fallback mode. Noise injection
// COM: via v_lshrrev_b32 12 is the anchor for the SR conversion path.

// SR-LABEL: <test_nosled_sr>:
// COM: --- Original site: branch replaces v_cvt_sr_fp8_f32 ---
// SR:       s_branch
// COM: --- Filler instruction preserved ---
// SR:       v_mov_b32{{.*}}0x43
// SR:       s_endpgm
// COM: --- Trampoline: NaN detection (anchor on v5) ---
// SR:       v_and_b32{{.*}}0x7fffffff, v5
// SR-NEXT:  v_cmp_lt_u32{{.*}}0x7f800000
// COM: --- Stochastic noise injection (anchor on v6) ---
// SR:       v_lshrrev_b32{{.*}}, 12, v6
// COM: --- F32 -> F16 -> UE5M3 ---
// SR:       v_cvt_f16_f32
// COM: --- Byte merge ---
// SR:       v_bfi_b32 v4,
// COM: --- VCC restore + branch back ---
// SR-NEXT:  s_mov_b32
// SR-NEXT:  s_branch

.globl test_nosled_sr
.p2align 8
.type test_nosled_sr,@function
test_nosled_sr:
  v_cvt_sr_fp8_f32 v4, v5, v6 clamp
  v_mov_b32 v7, 0x43
  s_endpgm
.Ltest_nosled_sr_end:
.size test_nosled_sr, .Ltest_nosled_sr_end-test_nosled_sr

// ---- Kernel 3: v_cvt_f32_fp8 CLAMP=1, no NOP sled ----------------------------
//
// COM: Unpack patch in trampoline fallback mode. Byte extraction via
// COM: v_and_b32 0xff, exp-31 F32 construction, F16 base path, and NaN
// COM: override with 0x7fa3d000.

// UNPACK-LABEL: <test_nosled_unpack>:
// COM: --- Original site: branch replaces v_cvt_f32_fp8 ---
// UNPACK:       s_branch
// COM: --- Filler instruction preserved ---
// UNPACK:       v_mov_b32{{.*}}0x44
// UNPACK:       s_endpgm
// COM: --- Trampoline: byte extraction (anchor on v9) ---
// UNPACK:       v_and_b32{{.*}}0xff, v9
// COM: --- NaN detection ---
// UNPACK-NEXT:  v_cmp_eq_u32{{.*}}0xff
// COM: --- F16 base path ---
// UNPACK:       v_cvt_f32_f16
// COM: --- NaN override (output to v8) ---
// UNPACK:       v_mov_b32{{.*}}0x7fa3d000
// UNPACK-NEXT:  v_cndmask_b32{{.*}}v8,
// COM: --- VCC restore + branch back ---
// UNPACK-NEXT:  s_mov_b32
// UNPACK-NEXT:  s_branch

.globl test_nosled_unpack
.p2align 8
.type test_nosled_unpack,@function
test_nosled_unpack:
  v_cvt_f32_fp8 v8, v9 clamp
  v_mov_b32 v10, 0x44
  s_endpgm
.Ltest_nosled_unpack_end:
.size test_nosled_unpack, .Ltest_nosled_unpack_end-test_nosled_unpack

.rodata
.p2align 8
.amdhsa_kernel test_nosled_pk
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.amdhsa_kernel test_nosled_sr
  .amdhsa_next_free_vgpr 8
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.amdhsa_kernel test_nosled_unpack
  .amdhsa_next_free_vgpr 11
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_nosled_pk
      .symbol: test_nosled_pk.kd
      .sgpr_count: 2
      .vgpr_count: 4
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: test_nosled_sr
      .symbol: test_nosled_sr.kd
      .sgpr_count: 2
      .vgpr_count: 8
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: test_nosled_unpack
      .symbol: test_nosled_unpack.kd
      .sgpr_count: 2
      .vgpr_count: 11
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
