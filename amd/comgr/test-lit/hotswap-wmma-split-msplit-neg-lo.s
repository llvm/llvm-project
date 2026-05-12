// Test M-split with neg_lo:[0,0,1] on src2.
//
// Differs from K-split: M-split has no carry, so the original src2's
// modifier applies to BOTH halves (each half has its own M-slice of
// src2). The MATRIX_FMT_FP4 modifiers added by the splitter come BEFORE
// the preserved neg_lo, mirroring how the printer orders them.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"

// DISASM-LABEL: <kernel>:
// DISASM-NOT:   v_wmma_f32_32x16x128_f4
// DISASM:       s_branch
// DISASM:       s_endpgm

// DISASM:       v_wmma_f32_16x16x128_f8f6f4 v[80:87], v[0:7], v[2:9], v[80:87] matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4 neg_lo:[0,0,1]
// DISASM-NEXT:  v_wmma_f32_16x16x128_f8f6f4 v[88:95], v[8:15], v[2:9], v[88:95] matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4 neg_lo:[0,0,1]
// DISASM-NEXT:  s_branch
.globl kernel
.p2align 8
.type kernel,@function
kernel:
  v_wmma_f32_32x16x128_f4 v[80:95], v[0:15], v[2:9], v[80:95] neg_lo:[0,0,1]
  s_endpgm
.size kernel, .-kernel

// Idempotency.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf
