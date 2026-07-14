// COM: HotSwap entry trampolines are supported across gfx125x.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1251 -nostdlib %s -o %t.gfx1251.elf
// RUN: hotswap-rewrite %t.gfx1251.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1251 amdgcn-amd-amdhsa--gfx1251 \
// RUN:   --entry-trampolines --output %t.gfx1251.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// RUN: %llvm-objdump -d %t.gfx1251.out.elf | %FileCheck --check-prefix=DISASM %s

// RUN: sed -e '/^\.amdgcn_target/d' \
// RUN:   -e 's/gfx1251/gfx12-5-generic/g' %s > %t.generic.s
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx12-5-generic -nostdlib \
// RUN:   %t.generic.s -o %t.generic.elf
// RUN: hotswap-rewrite %t.generic.elf \
// RUN:   amdgcn-amd-amdhsa--gfx12-5-generic \
// RUN:   amdgcn-amd-amdhsa--gfx12-5-generic \
// RUN:   --entry-trampolines --output %t.generic.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// RUN: %llvm-objdump -d %t.generic.out.elf | %FileCheck --check-prefix=DISASM %s

// API: RESULT: SUCCESS

// DISASM-LABEL: <entry_tramp_family_kernel>:
// DISASM: s_endpgm
// DISASM: global_wb
// DISASM-NEXT: v_nop
// DISASM-NEXT: s_get_pc_i64 s[2:3]
// DISASM-NEXT: s_add_co_u32 s2
// DISASM-NEXT: s_add_co_ci_u32 s3
// DISASM-NEXT: s_set_pc_i64 s[2:3]

.amdgcn_target "amdgcn-amd-amdhsa--gfx1251"
.text
.globl entry_tramp_family_kernel
.p2align 8
.type entry_tramp_family_kernel,@function
entry_tramp_family_kernel:
  v_mov_b32_e32 v0, 0
  s_endpgm
.Lentry_tramp_family_kernel_end:
.size entry_tramp_family_kernel, .Lentry_tramp_family_kernel_end-entry_tramp_family_kernel

.rodata
.p2align 8
.amdhsa_kernel entry_tramp_family_kernel
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 1
  .amdhsa_inst_pref_size 7
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: entry_tramp_family_kernel
      .symbol: entry_tramp_family_kernel.kd
      .sgpr_count: 1
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
