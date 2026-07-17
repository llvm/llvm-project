// COM: Test the true trampoline fallback path for ds_*_2addr_stride64_*
// COM: when no NOP sled is available. This file contains a single kernel
// COM: with no NOP padding, forcing emitReplacementCode to use
// COM: emitToTrampoline. The trampoline body (expanded DS instructions +
// COM: branch-back) is appended after .text via growWithTrampolines.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: Entry displacement is applied before ordinary instruction rewriting.
// COM: Verify that the direct entry prefix and appended DS trampoline coexist
// COM: with branch sites calculated from the displaced instruction layout.
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --entry-trampolines --output %t.entry.elf \
// RUN:   | %FileCheck --check-prefix=ENTRY-API %s
// ENTRY-API: RESULT: SUCCESS
// RUN: %llvm-objdump -d %t.entry.elf | %FileCheck --check-prefix=ENTRY %s
// ENTRY-LABEL: <test_ds_trampoline>:
// ENTRY-NEXT: global_wb
// ENTRY-NEXT: v_nop
// ENTRY-NEXT: s_branch
// ENTRY-NEXT: s_nop 0
// ENTRY-NEXT: s_wait_dscnt 0x0
// ENTRY-NEXT: s_endpgm
// ENTRY: ds_load_b32 v0, v2 offset:256
// ENTRY-NEXT: ds_load_b32 v1, v2 offset:768
// ENTRY-NEXT: s_wait_dscnt 0x0
// ENTRY-NEXT: s_branch
// ENTRY-NOT: s_get_pc_i64

// COM: The original DS2 is gone; s_branch forward replaces it. The drain
// COM: s_wait_dscnt stays at the original position with imm unchanged (0x0).
// DISASM-LABEL: <test_ds_trampoline>:
// DISASM-NOT: ds_load_2addr_stride64_b32
// DISASM: s_branch
// DISASM: s_wait_dscnt 0x0
// DISASM: s_endpgm

// COM: Trampoline body appended after .text: expanded DS loads + branch-back,
// COM: followed by an executable prefetch guard sized from inst_pref_size.
// DISASM: ds_load_b32 v0
// DISASM: ds_load_b32 v1
// DISASM: s_branch
// DISASM-NEXT: s_code_end
// DISASM-NEXT: s_code_end

// COM: Idempotency
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_ds_trampoline
.p2align 8
.type test_ds_trampoline,@function
test_ds_trampoline:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
  s_wait_dscnt 0x0
  s_endpgm
.Ltest_ds_trampoline_end:
.size test_ds_trampoline, .Ltest_ds_trampoline_end-test_ds_trampoline

.rodata
.p2align 8
.amdhsa_kernel test_ds_trampoline
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 1
  .amdhsa_inst_pref_size 2
.end_amdhsa_kernel
