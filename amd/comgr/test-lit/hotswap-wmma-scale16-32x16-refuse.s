// COM: Fail-closed gate for the 32x16 FP4 block-16 scaled WMMA. The M=32 form
// COM: needs an M-split on top of the K-split and has no exact lowering yet,
// COM: so the rewrite refuses it: any unlowerable v_wmma_scale16_f32_* makes the
// COM: rewrite return an error instead of emitting a wrong code object.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// COM: Default mode: presence of the unlowerable form fails the rewrite.
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR \
// RUN:   | %FileCheck --check-prefix=REFUSE %s
// REFUSE: RESULT: ERROR

// COM: Strict mode: same fail-closed result.
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --strict-mode --expect-status ERROR \
// RUN:   | %FileCheck --check-prefix=REFUSE %s

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

// --- Scale16 32x16 FP4 (block-16) -> refuse (fail closed) ---
.globl test_wmma_scale16_32x16
.p2align 8
.type test_wmma_scale16_32x16,@function
test_wmma_scale16_32x16:
  v_wmma_scale16_f32_32x16x128_f4 v[0:15], v[16:31], v[32:39], v[0:15], v[40:41], v[42:43]
  s_endpgm
.Ltest_wmma_scale16_32x16_end:
.size test_wmma_scale16_32x16, .Ltest_wmma_scale16_32x16_end-test_wmma_scale16_32x16

.rodata
.p2align 8
.amdhsa_kernel test_wmma_scale16_32x16
  .amdhsa_next_free_vgpr 44
  .amdhsa_next_free_sgpr 2
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
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
.end_amdgpu_metadata
