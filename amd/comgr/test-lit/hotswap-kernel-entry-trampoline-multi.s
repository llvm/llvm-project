// COM: A 16-byte direct prefix before the first kernel would misalign the
// COM: second kernel. HotSwap must retain the ABI's 256-byte entry alignment by
// COM: falling back to aligned appended stubs for the whole object.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --entry-trampolines --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// DISASM-LABEL: <entry_tramp_first>:
// DISASM-NEXT: v_mov_b32_e32 v0, 1
// DISASM-NEXT: s_endpgm
// DISASM-LABEL: <entry_tramp_second>:
// DISASM-NEXT: v_mov_b32_e32 v0, 2
// DISASM-NEXT: s_endpgm
// DISASM: global_wb // {{[0-9a-fA-F]+00}}:
// DISASM-NEXT: v_nop
// DISASM-NEXT: s_get_pc_i64
// DISASM: global_wb // {{[0-9a-fA-F]+00}}:
// DISASM-NEXT: v_nop
// DISASM-NEXT: s_get_pc_i64

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

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: entry_tramp_first
      .symbol: entry_tramp_first.kd
      .sgpr_count: 1
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: entry_tramp_second
      .symbol: entry_tramp_second.kd
      .sgpr_count: 1
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
