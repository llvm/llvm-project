// COM: Test the true trampoline fallback path for tensor_load_to_lds
// COM: when no NOP sled is available. Both live and dead descriptor variants
// COM: append the persistent s_pack_hh + tensor_load normalization via
// COM: growWithTrampolines.
// COM: Both force emitReplacementCode to use emitToTrampoline.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --strict-mode --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: Kernel 1 (dead SGPR, no in-function sled): original tensor_load
// COM: replaced by s_branch forward. Inter-function alignment padding is not a
// COM: borrowable sled, so trampoline bodies are appended after the original
// COM: functions.
// DISASM-LABEL: <test_tensor_trampoline>:
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_endpgm
// DISASM-NOT: tensor_load_to_lds

// COM: Kernel 2 (live SGPR, no in-function sled): the original tensor_load is
// COM: replaced by s_branch forward to its appended trampoline body.
// DISASM-LABEL: <test_tensor_trampoline_live>:
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_mov_b32
// DISASM-NEXT: s_endpgm
// DISASM-NOT: tensor_load_to_lds

// COM: Dead-SGPR trampoline body: s_pack_hh + tensor_load + branch-back,
// COM: appended in the trampoline pool section (a fresh vaddr above .text), so
// COM: objdump emits a section header between .text and the pool -- use DISASM
// COM: (not DISASM-NEXT) to cross that boundary.
// DISASM: s_pack_hh_b32_b16
// DISASM-NEXT: tensor_load_to_lds
// DISASM-NEXT: s_branch

// COM: Live-SGPR trampoline body (for kernel 2): the same persistent mask and
// COM: tensor sequence followed by branch-back.
// DISASM-NEXT: s_pack_hh_b32_b16 s4, 0, s4
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
.globl test_tensor_trampoline
.p2align 8
.type test_tensor_trampoline,@function
test_tensor_trampoline:
  tensor_load_to_lds s[0:3], s[4:11]
  s_endpgm
.Ltest_tensor_trampoline_end:
.size test_tensor_trampoline, .Ltest_tensor_trampoline_end-test_tensor_trampoline

// ---- Kernel 2: live SGPR, no NOP sled (persistent mask trampoline) --------

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
