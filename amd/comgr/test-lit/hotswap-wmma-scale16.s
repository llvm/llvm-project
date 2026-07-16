// COM: Test WMMA Scale16 (block-16 -> block-32) decomposition patch.
// COM:
// COM: Creates minimal gfx1250 code objects containing v_wmma_scale16_f32_*
// COM: instructions, runs the hotswap rewrite, and verifies the replacement
// COM: sequence covers: byte-pair max scale reduction followed by the
// COM: rewritten v_wmma_scale_f32_* instruction (VOP3PX2).

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// COM: -----------------------------------------------------------------------
// COM: Verify: Scale16 16x16 instruction is patched with scale reduction
// COM: preamble followed by the rewritten v_wmma_scale_f32_* instruction.
// COM:
// COM: Scale reduction per operand (13 instructions):
// COM:   4 byte pairs x (v_and_b32/v_bfe_u32 + v_bfe_u32/v_lshrrev_b32 +
// COM:                    v_max_u32 + optional v_lshl_or_b32)
// COM:
// COM: The preamble reduces block-16 scales from the B64 VGPR pairs
// COM: (v48:v49 for scale A, v50:v51 for scale B) into B32 scratch VGPRs
// COM: via byte-pair max, then the rewritten WMMA uses those scratch VGPRs.
// COM: -----------------------------------------------------------------------
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=SCALE16 %s

// SCALE16-LABEL: <test_wmma_scale16_16x16>:
// SCALE16:       s_branch
// COM: --- scale A reduction: byte pair 0 ---
// SCALE16:       v_and_b32{{.*}}0xff, v48
// SCALE16-NEXT:  v_bfe_u32{{.*}}v48, 8, 8
// SCALE16-NEXT:  v_max_u32
// COM: --- scale A reduction: byte pair 1 ---
// SCALE16-NEXT:  v_bfe_u32{{.*}}v48, 16, 8
// SCALE16-NEXT:  v_lshrrev_b32{{.*}}24, v48
// SCALE16-NEXT:  v_max_u32
// SCALE16-NEXT:  v_lshl_or_b32
// COM: --- scale A reduction: byte pair 2 ---
// SCALE16-NEXT:  v_and_b32{{.*}}0xff, v49
// SCALE16-NEXT:  v_bfe_u32{{.*}}v49, 8, 8
// SCALE16-NEXT:  v_max_u32
// SCALE16-NEXT:  v_lshl_or_b32
// COM: --- scale A reduction: byte pair 3 ---
// SCALE16-NEXT:  v_bfe_u32{{.*}}v49, 16, 8
// SCALE16-NEXT:  v_lshrrev_b32{{.*}}24, v49
// SCALE16-NEXT:  v_max_u32
// SCALE16-NEXT:  v_lshl_or_b32
// COM: --- scale B reduction: byte pair 0 ---
// SCALE16-NEXT:  v_and_b32{{.*}}0xff, v50
// SCALE16-NEXT:  v_bfe_u32{{.*}}v50, 8, 8
// SCALE16-NEXT:  v_max_u32
// COM: --- scale B reduction: byte pair 1 ---
// SCALE16-NEXT:  v_bfe_u32{{.*}}v50, 16, 8
// SCALE16-NEXT:  v_lshrrev_b32{{.*}}24, v50
// SCALE16-NEXT:  v_max_u32
// SCALE16-NEXT:  v_lshl_or_b32
// COM: --- scale B reduction: byte pair 2 ---
// SCALE16-NEXT:  v_and_b32{{.*}}0xff, v51
// SCALE16-NEXT:  v_bfe_u32{{.*}}v51, 8, 8
// SCALE16-NEXT:  v_max_u32
// SCALE16-NEXT:  v_lshl_or_b32
// COM: --- scale B reduction: byte pair 3 ---
// SCALE16-NEXT:  v_bfe_u32{{.*}}v51, 16, 8
// SCALE16-NEXT:  v_lshrrev_b32{{.*}}24, v51
// SCALE16-NEXT:  v_max_u32
// SCALE16-NEXT:  v_lshl_or_b32
// COM: --- rewritten WMMA (VOP3PX2): regular scale with scratch VGPRs ---
// SCALE16-NEXT:  v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7], v[16:31], v[32:47], v[0:7], v64, v65

// COM: -----------------------------------------------------------------------
// COM: Verify: Scale16 32x16 instruction is decomposed into two 16x16 WMMAs.
// COM:
// COM: The 32x16 FP4 instruction is B0-only (no A0 counterpart). It is split
// COM: along M into two 16x16 halves, each with scale reduction preamble and
// COM: rewritten v_wmma_scale_f32_16x16x128_f8f6f4 (VOP3PX2).
// COM: -----------------------------------------------------------------------
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=SPLIT32 %s

// SPLIT32-LABEL: <test_wmma_scale16_32x16>:
// SPLIT32:       s_branch
// COM: --- scale A reduction (shared by both halves) ---
// SPLIT32:       v_and_b32{{.*}}0xff, v40
// SPLIT32:       v_max_u32
// SPLIT32:       v_lshl_or_b32
// SPLIT32:       v_lshl_or_b32
// SPLIT32:       v_lshl_or_b32
// COM: --- scale B reduction ---
// SPLIT32:       v_and_b32{{.*}}0xff, v42
// SPLIT32:       v_max_u32
// SPLIT32:       v_lshl_or_b32
// SPLIT32:       v_lshl_or_b32
// SPLIT32:       v_lshl_or_b32
// COM: --- rewritten WMMA half 0 (rows 0-15) ---
// SPLIT32:       v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7], v[16:23], v[32:39], v[0:7], v48, v49
// COM: --- rewritten WMMA half 1 (rows 16-31) ---
// SPLIT32:       v_wmma_scale_f32_16x16x128_f8f6f4 v[8:15], v[24:31], v[32:39], v[8:15], v48, v49{{.*}}matrix_a_scale:MATRIX_SCALE_ROW1

// COM: -----------------------------------------------------------------------
// COM: Verify: 32x16 split preserves float inline-immediate src2.
// COM: -----------------------------------------------------------------------
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=SRC2FLOAT %s

// SRC2FLOAT-LABEL: <test_wmma_scale16_32x16_src2_neg_half>:
// SRC2FLOAT:       s_branch
// SRC2FLOAT:       v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7], v[16:23], v[32:39], -0.5, v48, v49
// SRC2FLOAT:       v_wmma_scale_f32_16x16x128_f8f6f4 v[8:15], v[24:31], v[32:39], -0.5, v48, v49{{.*}}matrix_a_scale:MATRIX_SCALE_ROW1

// COM: -----------------------------------------------------------------------
// COM: Verify: regular Scale instruction is NOT patched.
// COM: -----------------------------------------------------------------------
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=NOPATCH %s

// NOPATCH-LABEL: <test_wmma_scale_16x16>:
// NOPATCH-NEXT:  v_wmma_scale_f32_16x16x128_f8f6f4

// COM: -----------------------------------------------------------------------
// COM: Idempotency: running the rewrite a second time should succeed and
// COM: produce the same output.
// COM: -----------------------------------------------------------------------
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=IDEMP %s
// IDEMP: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

// --- Kernel 1: Scale16 (should be patched) ---
.globl test_wmma_scale16_16x16
.p2align 8
.type test_wmma_scale16_16x16,@function
test_wmma_scale16_16x16:
  v_wmma_scale16_f32_16x16x128_f8f6f4 v[0:7], v[16:31], v[32:47], v[0:7], v[48:49], v[50:51]
  s_endpgm
.Ltest_wmma_scale16_16x16_end:
.size test_wmma_scale16_16x16, .Ltest_wmma_scale16_16x16_end-test_wmma_scale16_16x16

// --- Kernel 2: Scale16 32x16 FP4 (should be split into two 16x16) ---
.globl test_wmma_scale16_32x16
.p2align 8
.type test_wmma_scale16_32x16,@function
test_wmma_scale16_32x16:
  v_wmma_scale16_f32_32x16x128_f4 v[0:15], v[16:31], v[32:39], v[0:15], v[40:41], v[42:43]
  s_endpgm
.Ltest_wmma_scale16_32x16_end:
.size test_wmma_scale16_32x16, .Ltest_wmma_scale16_32x16_end-test_wmma_scale16_32x16

// --- Kernel 3: Scale16 32x16 with float inline src2 (should preserve src2) ---
.globl test_wmma_scale16_32x16_src2_neg_half
.p2align 8
.type test_wmma_scale16_32x16_src2_neg_half,@function
test_wmma_scale16_32x16_src2_neg_half:
  v_wmma_scale16_f32_32x16x128_f4 v[0:15], v[16:31], v[32:39], -0.5, v[40:41], v[42:43]
  s_endpgm
.Ltest_wmma_scale16_32x16_src2_neg_half_end:
.size test_wmma_scale16_32x16_src2_neg_half, .Ltest_wmma_scale16_32x16_src2_neg_half_end-test_wmma_scale16_32x16_src2_neg_half

// --- Kernel 4: regular Scale (should NOT be patched) ---
.globl test_wmma_scale_16x16
.p2align 8
.type test_wmma_scale_16x16,@function
test_wmma_scale_16x16:
  v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7], v[16:31], v[32:47], v[0:7], v48, v50
  s_endpgm
.Ltest_wmma_scale_16x16_end:
.size test_wmma_scale_16x16, .Ltest_wmma_scale_16x16_end-test_wmma_scale_16x16

.rodata
.p2align 8
.amdhsa_kernel test_wmma_scale16_16x16
  .amdhsa_next_free_vgpr 52
  .amdhsa_next_free_sgpr 2
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel
.amdhsa_kernel test_wmma_scale16_32x16
  .amdhsa_next_free_vgpr 44
  .amdhsa_next_free_sgpr 2
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel
.amdhsa_kernel test_wmma_scale16_32x16_src2_neg_half
  .amdhsa_next_free_vgpr 44
  .amdhsa_next_free_sgpr 2
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel
.amdhsa_kernel test_wmma_scale_16x16
  .amdhsa_next_free_vgpr 52
  .amdhsa_next_free_sgpr 2
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_wmma_scale16_16x16
      .symbol: test_wmma_scale16_16x16.kd
      .sgpr_count: 2
      .vgpr_count: 52
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
    - .name: test_wmma_scale16_32x16
      .symbol: test_wmma_scale16_32x16.kd
      .sgpr_count: 2
      .vgpr_count: 44
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
    - .name: test_wmma_scale16_32x16_src2_neg_half
      .symbol: test_wmma_scale16_32x16_src2_neg_half.kd
      .sgpr_count: 2
      .vgpr_count: 44
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
    - .name: test_wmma_scale_16x16
      .symbol: test_wmma_scale_16x16.kd
      .sgpr_count: 2
      .vgpr_count: 52
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
