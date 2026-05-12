// Test K-split with neg_hi:[0,0,1] on src2 -- same per-half behavior
// as neg_lo (preserved on first half, dropped on second half) but for
// the NEG_HI bit (SISrcMods::NEG_HI = 1 << 1, vs NEG = 1 << 0) which
// the splitter projects onto the same modifier-suffix synthesis path.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"

// DISASM-LABEL: <kernel>:
// DISASM-NOT:   v_wmma_f16_16x16x128_fp8_bf8
// DISASM:       s_branch
// DISASM:       s_endpgm

// COM: Operand-shape note: src0/src1 disjoint from dst per @earlyclobber $vdst.
// DISASM:       v_wmma_f16_16x16x64_fp8_bf8 v[32:35], v[0:7], v[16:23], v[32:35] neg_hi:[0,0,1]
// DISASM-NEXT:  v_wmma_f16_16x16x64_fp8_bf8 v[32:35], v[8:15], v[24:31], v[32:35]{{[[:space:]]*\/\/}}
// DISASM-NEXT:  s_branch
.globl kernel
.p2align 8
.type kernel,@function
kernel:
  v_wmma_f16_16x16x128_fp8_bf8 v[32:35], v[0:15], v[16:31], v[32:35] neg_hi:[0,0,1]
  s_endpgm
.size kernel, .-kernel

// Idempotency.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf
