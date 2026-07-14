// COM: Test persistent tensor_load_to_lds descriptor normalization across a
// COM: control-flow edge. The descriptor remains masked on either successor,
// COM: so the rewrite needs no liveness-dependent save/restore sequence.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --strict-mode --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: Kernel 1 (branch guard): s_cbranch_scc1 sits between tensor_load and
// COM: s_mov (which reads s4). The later read sees the normalized descriptor.
// DISASM-LABEL: <test_tensor_branch_guard>:
// DISASM: s_branch
// DISASM: s_cbranch_scc1
// DISASM: s_endpgm
// DISASM: s_pack_hh_b32_b16 s4, 0, s4
// DISASM-NEXT: tensor_load_to_lds
// DISASM-NEXT: s_branch

// COM: Idempotency
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --strict-mode --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_branch_guard
.p2align 8
.type test_tensor_branch_guard,@function
test_tensor_branch_guard:
  tensor_load_to_lds s[0:3], s[4:11]
  s_cbranch_scc1 .Lskip
  s_mov_b32 s0, s4
.Lskip:
  s_endpgm
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
.Ltest_tensor_branch_guard_end:
.size test_tensor_branch_guard, .Ltest_tensor_branch_guard_end-test_tensor_branch_guard

.rodata
.p2align 8
.amdhsa_kernel test_tensor_branch_guard
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel
