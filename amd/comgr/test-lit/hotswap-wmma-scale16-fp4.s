// COM: Block-16 scaled WMMA (16x16x128) with FP4 matrix data. Unlike FP8, an
// COM: FP4 K=32 block sits in one lane group and its low/high-16 subblocks
// COM: split along the VGPR index, so the masked-A copy nulls the opposite
// COM: subblock's VGPRs (v_mov ..., 0) instead of using a lane mask.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=RESULT %s
// RESULT: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-NOT: v_wmma_scale16
// COM: pass-low keeps the low-16 subblock VGPRs and zeros the high-16, then a
// COM: block-32 WMMA with the even-byte scale gather writes v[0:7]. FP4 uses a
// COM: VGPR select, never a lane mask, so no v_cndmask appears in either pass.
// DISASM-NOT: v_cndmask_b32_e64
// DISASM: v_mov_b32{{(_e32)?}} v{{[0-9]+}}, 0
// DISASM: v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7], v[{{[0-9]+}}:{{[0-9]+}}], v[32:39], v[0:7],{{.*}}matrix_a_fmt:MATRIX_FMT_FP4
// COM: exactly one gfx1250 hazard v_nop before the pass-high masked-A VALU.
// DISASM-COUNT-1: v_nop
// DISASM-NEXT: v_mov_b32{{(_e32)?}} v{{[0-9]+}}, 0
// DISASM-NOT: v_cndmask_b32_e64
// COM: pass-high keeps the high-16 subblock VGPRs, odd-byte scale gather,
// COM: accumulating onto pass-low through v[0:7].
// DISASM: v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7], v[{{[0-9]+}}:{{[0-9]+}}], v[32:39], v[0:7],{{.*}}matrix_a_fmt:MATRIX_FMT_FP4

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

// --- Scale16 16x16 (block-16) FP4 -> exact VGPR-select K-split ---
.globl test_wmma_scale16_16x16_fp4
.p2align 8
.type test_wmma_scale16_16x16_fp4,@function
test_wmma_scale16_16x16_fp4:
  v_wmma_scale16_f32_16x16x128_f8f6f4 v[0:7], v[16:23], v[32:39], v[0:7], v[48:49], v[50:51] matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4
  s_endpgm
.Ltest_wmma_scale16_16x16_fp4_end:
.size test_wmma_scale16_16x16_fp4, .Ltest_wmma_scale16_16x16_fp4_end-test_wmma_scale16_16x16_fp4

.rodata
.p2align 8
.amdhsa_kernel test_wmma_scale16_16x16_fp4
  .amdhsa_next_free_vgpr 52
  .amdhsa_next_free_sgpr 2
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_wmma_scale16_16x16_fp4
      .symbol: test_wmma_scale16_16x16_fp4.kd
      .sgpr_count: 2
      .vgpr_count: 52
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
