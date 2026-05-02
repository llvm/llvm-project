// COM: Passthrough test for the VOP3PX2 scale_src2 bit-field fix. A kernel
// COM: with no V_WMMA_SCALE* instructions must be left structurally
// COM: unchanged: no bits are modified, and the disassembly must match the
// COM: original layout.

// RUN: %clang --target=amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// COM: No V_WMMA_SCALE instructions, so the patch must not fire.
// COM: Verify the disassembly layout is preserved and that v_wmma_scale
// COM: does not appear (DISASM-NOT scope: between v_wmma_f32 and s_endpgm).
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM: v_wmma_f32_16x16x128_f8f6f4
// DISASM-NOT: v_wmma_scale
// DISASM: s_endpgm

// COM: Idempotency: second rewrite must produce identical bytes.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_vop3px2_noop
.p2align 8
.type test_vop3px2_noop,@function
test_vop3px2_noop:
  // Regular (non-scale) WMMA: patch must not touch this.
  v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[8:23], v[24:35], v[0:7] matrix_a_fmt:MATRIX_FMT_BF8 matrix_b_fmt:MATRIX_FMT_FP6
  s_endpgm
.Ltest_vop3px2_noop_end:
.size test_vop3px2_noop, .Ltest_vop3px2_noop_end-test_vop3px2_noop

.rodata
.p2align 8
.amdhsa_kernel test_vop3px2_noop
  .amdhsa_next_free_vgpr 36
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
