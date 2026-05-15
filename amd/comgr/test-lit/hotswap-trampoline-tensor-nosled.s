// COM: Test the true trampoline fallback path for tensor_load_to_lds
// COM: when no NOP sled is available. Two variants:
// COM:   dead SGPR — s_pack_hh + tensor_load appended via growWithTrampolines
// COM:   live SGPR — save/pack/tensor/restore (4-instruction sequence)
// COM:              appended via growWithTrampolines, the largest replacement
// COM: Both force emitReplacementCode to use emitToTrampoline.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: Kernel 1 (dead SGPR, no sled): original tensor_load replaced by
// COM: s_branch forward. Trampoline body appended in alignment padding.
// DISASM-LABEL: <test_tensor_trampoline>:
// DISASM-NOT: tensor_load_to_lds
// DISASM: s_branch
// DISASM: s_endpgm

// COM: Dead-SGPR trampoline body: s_pack_hh + tensor_load + branch-back.
// DISASM: s_pack_hh_b32_b16
// DISASM: tensor_load_to_lds
// DISASM: s_branch

// COM: Live-SGPR trampoline body (for kernel 2): also placed in the
// COM: padding region. save + pack + tensor + restore + branch-back.
// DISASM: v_writelane_b32
// DISASM: s_pack_hh_b32_b16
// DISASM: tensor_load_to_lds
// DISASM: v_readlane_b32
// DISASM: s_branch

// COM: Kernel 2 (live SGPR, no sled): the original tensor_load is
// COM: replaced by s_branch backward to the trampoline body above.
// DISASM-LABEL: <test_tensor_trampoline_live>:
// DISASM-NOT: tensor_load_to_lds
// DISASM: s_branch
// DISASM: s_mov_b32
// DISASM: s_endpgm

// COM: Idempotency
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_trampoline
.p2align 8
.type test_tensor_trampoline,@function
test_tensor_trampoline:
  tensor_load_to_lds s[0:3], s[4:11]
  s_endpgm
.Ltest_tensor_trampoline_end:
.size test_tensor_trampoline, .Ltest_tensor_trampoline_end-test_tensor_trampoline

// ---- Kernel 2: live SGPR, no NOP sled (trampoline + save/restore) ----------

.globl test_tensor_trampoline_live
.p2align 8
.type test_tensor_trampoline_live,@function
test_tensor_trampoline_live:
  tensor_load_to_lds s[0:3], s[4:11]
  s_mov_b32 s0, s4
  s_endpgm
.Ltest_tensor_trampoline_live_end:
.size test_tensor_trampoline_live, .Ltest_tensor_trampoline_live_end-test_tensor_trampoline_live

.rodata
.p2align 8
.amdhsa_kernel test_tensor_trampoline
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.p2align 8
.amdhsa_kernel test_tensor_trampoline_live
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel
