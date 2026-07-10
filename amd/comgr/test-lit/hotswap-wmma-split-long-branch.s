// COM: HSV-009 / PLAT-205406: WMMA-split shares emitToTrampoline with the other
// COM: patch families, so a split site beyond s_branch's +-128 KB reach of the
// COM: appended pool takes the far path -- which on gfx1250 A0 is DECLINED
// COM: (left unpatched) because the backward s_add_pc_i64 branch-back corrupts
// COM: wave state at runtime. The original v_wmma_f32_16x16x128_fp8_fp8 stays at
// COM: the site; it is neither split into two K=64 halves nor redirected. A
// COM: large .rept filler (~160 KB, non-NOP) forces the far case.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: Declined: the site keeps its original K=128 WMMA (it is NOT overwritten
// COM: with an s_add_pc_i64 forward branch), and no split halves
// COM: (v_wmma_f32_16x16x64_fp8_fp8) or long-branch redirect are emitted.
// DISASM-LABEL: <test_wsplit_far>:
// DISASM-NEXT: v_wmma_f32_16x16x128_fp8_fp8
// DISASM-NOT: s_add_pc_i64
// DISASM-NOT: v_wmma_f32_16x16x64_fp8_fp8

// COM: Idempotency: rewriting the output again must be a no-op.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_wsplit_far
.p2align 8
.type test_wsplit_far,@function
test_wsplit_far:
  v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[0:15], v[16:31], v[32:39]
  s_endpgm
  // ~160 KB of non-NOP filler so the appended trampoline pool is beyond
  // s_branch's +-128 KB reach from the WMMA above (forces the long-branch path).
  .rept 40000
    s_mov_b32 s0, s1
  .endr
.size test_wsplit_far, .-test_wsplit_far
