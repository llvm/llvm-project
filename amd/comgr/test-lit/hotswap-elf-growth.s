// COM: Verify that hotswap-rewrite succeeds on a clang-assembled ELF where
// COM: .dynamic follows .text (the layout that previously caused
// COM: growWithTrampolines to refuse). No patches are applied here (the
// COM: kernel contains no patchable instructions), but the rewrite pipeline
// COM: must accept the ELF rather than returning an error.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// COM: Confirm .dynamic exists after .text in the input ELF.
// RUN: %llvm-readelf --section-headers %t.elf | %FileCheck --check-prefix=LAYOUT %s
// LAYOUT: .text
// LAYOUT: .dynamic

// COM: hotswap-rewrite must succeed (not reject the ELF).
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// COM: Output ELF is valid and disassemblable.
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM: file format elf64-amdgpu
// DISASM: s_endpgm

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_elf_growth
.p2align 8
.type test_elf_growth,@function
test_elf_growth:
  v_mov_b32_e32 v0, 0
  s_endpgm
.Ltest_elf_growth_end:
.size test_elf_growth, .Ltest_elf_growth_end-test_elf_growth

.rodata
.p2align 8
.amdhsa_kernel test_elf_growth
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel
