// COM: Test multiple FP8 E5M3 patch sites in a single kernel.
// COM:
// COM: Exercises: stacking of multiple trampolines, cross-instruction-type
// COM: coexistence, overlapping src/dst registers.
// COM:
// COM: Companion tests:
// COM:   hotswap-cvt-pk-fp8.s   — v_cvt_pk_fp8_f32  (pack F32->E5M3)
// COM:   hotswap-cvt-sr-fp8.s   — v_cvt_sr_fp8_f32  (stochastic round F32->E5M3)
// COM:   hotswap-cvt-f32-fp8.s  — v_cvt_f32_fp8     (unpack E5M3->F32)

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --dump %t.out.elf --check-idempotent \
// RUN:   | %FileCheck --check-prefix=API %s
// API: REWRITE: SUCCESS
// API: IDEMPOTENT: YES

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=SAME %s
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=MIXED %s
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=OVERLAP %s

// ---- Kernel 1: two v_cvt_pk_fp8_f32 clamp sites (same type) ------------------
//
// COM: Both pk instructions must be replaced by s_branch, and both trampolines
// COM: must appear in the output. Tests trampoline stacking for identical
// COM: instruction types.

// SAME-LABEL: <test_multi_fp8_same>:
// SAME-NOT:   v_cvt_pk_fp8_f32
// SAME:       s_branch
// SAME-NOT:   v_cvt_pk_fp8_f32
// SAME:       s_branch
// SAME:       s_endpgm
// COM: --- First pk trampoline (vdst=v0, src0=v1, src1=v2) ---
// SAME:       v_and_b32{{.*}}0x7fffffff, v1
// SAME:       v_lshl_or_b32
// SAME-NEXT:  v_bfi_b32 v0,
// COM: --- Second pk trampoline (vdst=v4, src0=v5, src1=v6) ---
// SAME:       v_and_b32{{.*}}0x7fffffff, v5
// SAME:       v_lshl_or_b32
// SAME-NEXT:  v_bfi_b32 v4,

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_multi_fp8_same
.p2align 8
.type test_multi_fp8_same,@function
test_multi_fp8_same:
  v_cvt_pk_fp8_f32 v0, v1, v2 clamp
  v_mov_b32 v3, 0
  v_cvt_pk_fp8_f32 v4, v5, v6 clamp
  s_endpgm
.Ltest_multi_fp8_same_end:
.size test_multi_fp8_same, .Ltest_multi_fp8_same_end-test_multi_fp8_same

// ---- Kernel 2: all three FP8 types in one kernel (mixed) ---------------------
//
// COM: Each instruction type gets its own s_branch + trampoline with a distinct
// COM: signature: pk uses v_lshl_or_b32, sr uses v_lshrrev_b32 12 for noise
// COM: injection, unpack uses v_or_b32 0x47800000 for exp-31 construction.

// MIXED-LABEL: <test_multi_fp8_mixed>:
// MIXED-NOT:  v_cvt_pk_fp8_f32
// MIXED:      s_branch
// MIXED-NOT:  v_cvt_sr_fp8_f32
// MIXED:      s_branch
// MIXED-NOT:  v_cvt_f32_fp8
// MIXED:      s_branch
// MIXED:      s_endpgm
// COM: --- pk trampoline: pack via v_lshl_or_b32 + merge into v0 ---
// MIXED:      v_and_b32{{.*}}0x7fffffff, v1
// MIXED:      v_lshl_or_b32
// MIXED-NEXT: v_bfi_b32 v0,
// COM: --- sr trampoline: noise injection via v_lshrrev_b32 12 ---
// MIXED:      v_lshrrev_b32{{.*}}, 12, v6
// MIXED:      v_bfi_b32 v4,
// COM: --- unpack trampoline: byte extraction + exp-31 F32 construction ---
// MIXED:      v_and_b32{{.*}}0xff, v9
// MIXED:      v_or_b32{{.*}}0x47800000
// MIXED:      v_cvt_f32_f16

.globl test_multi_fp8_mixed
.p2align 8
.type test_multi_fp8_mixed,@function
test_multi_fp8_mixed:
  v_cvt_pk_fp8_f32 v0, v1, v2 clamp
  v_mov_b32 v3, 0
  v_cvt_sr_fp8_f32 v4, v5, v6 clamp
  v_mov_b32 v7, 0
  v_cvt_f32_fp8 v8, v9 clamp
  s_endpgm
.Ltest_multi_fp8_mixed_end:
.size test_multi_fp8_mixed, .Ltest_multi_fp8_mixed_end-test_multi_fp8_mixed

// ---- Kernel 3: overlapping src/dst registers (vdst==src0) --------------------
//
// COM: v_cvt_pk_fp8_f32 v0, v0, v1 — vdst overlaps src0. Verifies the patch
// COM: is not skipped due to register overlap.

// OVERLAP-LABEL: <test_multi_fp8_overlap>:
// OVERLAP-NOT:   v_cvt_pk_fp8_f32
// OVERLAP:       s_branch
// COM: --- Trampoline applied despite vdst==src0 overlap ---
// OVERLAP:       v_and_b32{{.*}}0x7fffffff, v0
// OVERLAP:       v_bfi_b32 v0,

.globl test_multi_fp8_overlap
.p2align 8
.type test_multi_fp8_overlap,@function
test_multi_fp8_overlap:
  v_cvt_pk_fp8_f32 v0, v0, v1 clamp
  s_endpgm
.Ltest_multi_fp8_overlap_end:
.size test_multi_fp8_overlap, .Ltest_multi_fp8_overlap_end-test_multi_fp8_overlap

.rodata
.p2align 8
.amdhsa_kernel test_multi_fp8_same
  .amdhsa_next_free_vgpr 7
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.amdhsa_kernel test_multi_fp8_mixed
  .amdhsa_next_free_vgpr 10
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.amdhsa_kernel test_multi_fp8_overlap
  .amdhsa_next_free_vgpr 2
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_multi_fp8_same
      .symbol: test_multi_fp8_same.kd
      .sgpr_count: 2
      .vgpr_count: 7
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: test_multi_fp8_mixed
      .symbol: test_multi_fp8_mixed.kd
      .sgpr_count: 2
      .vgpr_count: 10
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: test_multi_fp8_overlap
      .symbol: test_multi_fp8_overlap.kd
      .sgpr_count: 2
      .vgpr_count: 2
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
