// Test K-split src2 = FP inline immediate (e.g. `1.0`). Must be preserved
// through the printer round-trip rather than reformatted as itostr() --
// `1.0` and integer `1` encode at distinct VOP3P inline-const slots
// (242 vs 1 per the AMDGPU ISA), so emitting `1` would change the
// instruction.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"

// COM: Source: K=128 opcode replaced by s_branch into trampoline.
// DISASM-LABEL: <kernel>:
// DISASM-NOT:   v_wmma_f32_16x16x128_fp8_fp8
// DISASM:       s_branch
// DISASM:       s_endpgm

// COM: First half preserves the FP imm `1.0` verbatim (printer round-trip).
// COM: Second half's src2 becomes the dst register (carry).
// COM: Operand-shape note: src0/src1 disjoint from dst per @earlyclobber $vdst.
// DISASM:       v_wmma_f32_16x16x64_fp8_fp8 v[32:39], v[0:7], v[16:23], 1.0
// DISASM-NEXT:  v_wmma_f32_16x16x64_fp8_fp8 v[32:39], v[8:15], v[24:31], v[32:39]
// DISASM-NEXT:  s_branch
.globl kernel
.p2align 8
.type kernel,@function
kernel:
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], 1.0
  s_endpgm
.size kernel, .-kernel

// Idempotency.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf
