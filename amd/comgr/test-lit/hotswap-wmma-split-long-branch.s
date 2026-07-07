// COM: WMMA-split over the s_add_pc_i64 long-branch path. WMMA-split shares
// COM: emitToTrampoline with the other patch families, so a split site beyond
// COM: s_branch's +-128 KB reach of the appended pool uses an s_add_pc_i64
// COM: long branch on both edges instead of falling back. A large .rept filler
// COM: (~160 KB, non-NOP) pushes the pool past s_branch's real reach from the
// COM: WMMA at the top.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: Forward edge: the 8-byte WMMA site is overwritten in place with an
// COM: 8-byte s_add_pc_i64 long branch (NOT an s_branch).
// DISASM-LABEL: <test_wsplit_far>:
// DISASM: s_add_pc_i64
// DISASM-NEXT: s_endpgm

// COM: Trampoline body: the two K=64 split halves, then an s_add_pc_i64
// COM: long branch-back.
// DISASM: v_wmma_f32_16x16x64_fp8_fp8
// DISASM-NEXT: v_wmma_f32_16x16x64_fp8_fp8
// DISASM-NEXT: s_add_pc_i64

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
