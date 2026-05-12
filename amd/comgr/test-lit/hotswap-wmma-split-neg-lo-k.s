// Test K-split with neg_lo:[0,0,1] on src2: the modifier negates the
// src2 input bit. On a K-split:
//   - First half: src2 IS the original input -> the modifier applies.
//   - Second half: src2 := dst (the partial-product carry from the
//     first half) -> the modifier MUST be cleared (negating the partial
//     product would subtract the previously-accumulated value, yielding
//     `D = A_hi*B_hi - D_partial`, which is wrong).
//
// neg_lo:[0,0,0] is the printer's omitted-default form, so the second
// half ends up with no neg_lo suffix at all.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"

// DISASM-LABEL: <kernel>:
// DISASM-NOT:   v_wmma_f32_16x16x128_bf8_bf8
// DISASM:       s_branch
// DISASM:       s_endpgm

// COM: First half preserves neg_lo:[0,0,1]; second half emits no
// COM: modifier suffix (the src2 bit was cleared and the resulting
// COM: all-zero modifier vector is the printer's default which is
// COM: omitted). The trailing `//` comment is what immediately follows
// COM: the operand list when no modifier suffix is emitted.
// DISASM:       v_wmma_f32_16x16x64_bf8_bf8 v[24:31], v[0:7], v[8:15], v[24:31] neg_lo:[0,0,1]
// DISASM-NEXT:  v_wmma_f32_16x16x64_bf8_bf8 v[24:31], v[8:15], v[16:23], v[24:31]{{[[:space:]]*\/\/}}
// DISASM-NEXT:  s_branch
.globl kernel
.p2align 8
.type kernel,@function
kernel:
  v_wmma_f32_16x16x128_bf8_bf8 v[24:31], v[0:15], v[8:23], v[24:31] neg_lo:[0,0,1]
  s_endpgm
.size kernel, .-kernel

// Idempotency.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf
