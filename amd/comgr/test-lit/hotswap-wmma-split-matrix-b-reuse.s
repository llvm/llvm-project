// Test that matrix_b_reuse is stripped on both halves of a K-split,
// same rationale as matrix_a_reuse (the B matrix is sliced in half by
// K, so the reuse-buffer assertion no longer holds after rewrite).

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"

// DISASM-LABEL: <kernel>:
// DISASM-NOT:   v_wmma_f32_16x16x128_bf8_fp8
// DISASM:       s_branch
// DISASM:       s_endpgm

// DISASM:       v_wmma_f32_16x16x64_bf8_fp8 v[48:55], v[0:7], v[8:15], v[48:55]{{[[:space:]]*\/\/}}
// DISASM-NEXT:  v_wmma_f32_16x16x64_bf8_fp8 v[48:55], v[8:15], v[16:23], v[48:55]{{[[:space:]]*\/\/}}
// DISASM-NEXT:  s_branch

// DISASM-NOT:   v_wmma_f32_16x16x64_bf8_fp8 v[48:55]{{.*}}matrix_b_reuse
.globl kernel
.p2align 8
.type kernel,@function
kernel:
  v_wmma_f32_16x16x128_bf8_fp8 v[48:55], v[0:15], v[8:23], v[48:55] matrix_b_reuse
  s_endpgm
.size kernel, .-kernel

// Idempotency.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf
