// Test M-split src2 = FP inline immediate (`1.0`).
//
// Differs from K-split FP-imm: there is no carry between halves on
// the M axis (each half writes a different M-slice of dst), so BOTH
// halves carry the same src2 imm. Both halves also carry the
// splitter-added `matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4`
// suffix (the destination opcode v_wmma_f32_16x16x128_f8f6f4 has
// matrix_*_fmt operands that the source opcode v_wmma_f32_32x16x128_f4
// does not, and they must be set to MATRIX_FMT_FP4 so the f8f6f4
// destination interprets the data as the source's f4 layout).

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

// DISASM:       v_wmma_f32_16x16x128_f8f6f4 v[64:71], v[0:7], v[2:9], 1.0 matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4
// DISASM-NEXT:  v_wmma_f32_16x16x128_f8f6f4 v[72:79], v[8:15], v[2:9], 1.0 matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4
// DISASM-NEXT:  s_branch
.globl kernel
.p2align 8
.type kernel,@function
kernel:
  v_wmma_f32_32x16x128_f4 v[64:79], v[0:15], v[2:9], 1.0
  s_endpgm
.size kernel, .-kernel

// Idempotency.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf
