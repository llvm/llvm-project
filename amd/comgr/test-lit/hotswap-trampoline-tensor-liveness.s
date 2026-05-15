// COM: Test isSgprLiveAfter edge cases for tensor_load_to_lds patching.
// COM: A branch instruction between the tensor_load and the next use of
// COM: the descriptor SGPR forces the heuristic to conservatively assume
// COM: the SGPR is live, producing save/restore even though the use may
// COM: not execute on all paths.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: Kernel 1 (branch guard): s_cbranch_scc1 sits between tensor_load
// COM: and s_mov (which reads s4). isSgprLiveAfter returns true at the
// COM: branch, so save/restore is emitted conservatively.
// DISASM-LABEL: <test_tensor_branch_guard>:
// DISASM: s_branch
// DISASM: s_cbranch_scc1
// DISASM: v_writelane_b32
// DISASM: s_pack_hh_b32_b16
// DISASM: tensor_load_to_lds
// DISASM: v_readlane_b32
// DISASM: s_branch

// COM: Idempotency
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
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
