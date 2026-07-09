// COM: The entry-trampoline rewrite must redirect every kernel descriptor in
// COM: the code object, not just the first one. Each descriptor gets its own
// COM: appended PC-relative entry stub.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --entry-trampolines --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// DISASM-LABEL: <entry_tramp_first>:
// DISASM: s_endpgm
// DISASM-LABEL: <entry_tramp_second>:
// DISASM: s_endpgm
// DISASM: global_wb
// DISASM-NEXT: v_nop
// DISASM-NEXT: s_get_pc_i64 s[8:9]
// DISASM-NEXT: s_add_co_u32 s8
// DISASM-NEXT: s_add_co_ci_u32 s9
// DISASM-NEXT: s_set_pc_i64 s[8:9]
// DISASM: global_wb
// DISASM-NEXT: v_nop
// DISASM-NEXT: s_get_pc_i64 s[8:9]
// DISASM-NEXT: s_add_co_u32 s8
// DISASM-NEXT: s_add_co_ci_u32 s9
// DISASM-NEXT: s_set_pc_i64 s[8:9]

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl entry_tramp_first
.p2align 8
.type entry_tramp_first,@function
entry_tramp_first:
  v_mov_b32_e32 v0, 1
  s_endpgm
.Lentry_tramp_first_end:
.size entry_tramp_first, .Lentry_tramp_first_end-entry_tramp_first

.globl entry_tramp_second
.p2align 8
.type entry_tramp_second,@function
entry_tramp_second:
  v_mov_b32_e32 v0, 2
  s_endpgm
.Lentry_tramp_second_end:
.size entry_tramp_second, .Lentry_tramp_second_end-entry_tramp_second

.rodata
.p2align 8
.amdhsa_kernel entry_tramp_first
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.p2align 8
.amdhsa_kernel entry_tramp_second
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel
