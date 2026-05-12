// Test K-split src2 = integer 0 inline immediate. This is the
// canonical compiler-folded zero accumulator (`acc = {0,...,0}` ->
// inline-const slot 1). Verify the imm is preserved on the first
// half and the second half uses dst as the carry.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"

// DISASM-LABEL: <kernel>:
// DISASM-NOT:   v_wmma_f32_16x16x128_fp8_fp8
// DISASM:       s_branch
// DISASM:       s_endpgm

// COM: Operand-shape note: src0/src1 disjoint from dst per @earlyclobber $vdst.
// DISASM:       v_wmma_f32_16x16x64_fp8_fp8 v[32:39], v[0:7], v[16:23], 0
// DISASM-NEXT:  v_wmma_f32_16x16x64_fp8_fp8 v[32:39], v[8:15], v[24:31], v[32:39]
// DISASM-NEXT:  s_branch
.globl kernel
.p2align 8
.type kernel,@function
kernel:
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 0
  s_endpgm
.size kernel, .-kernel

// Idempotency.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf
