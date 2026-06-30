// COM: HotSwap redirects kernel descriptors to appended PC-relative entry
// COM: stubs when the entry-trampoline flag is explicitly enabled.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.default.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// RUN: cmp %t.elf %t.default.elf
// RUN: %llvm-objdump -d %t.default.elf | %FileCheck --check-prefix=NO-TRAMP %s
// NO-TRAMP-LABEL: <entry_tramp_kernel>:
// NO-TRAMP: s_endpgm
// NO-TRAMP-NOT: global_wb

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --entry-trampolines --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// RUN: %llvm-readelf --notes %t.out.elf | %FileCheck --check-prefix=METADATA %s

// DISASM-LABEL: <entry_tramp_kernel>:
// DISASM: s_endpgm
// DISASM: global_wb
// DISASM-NEXT: v_nop
// DISASM-NEXT: s_get_pc_i64 s[8:9]
// DISASM-NEXT: s_add_co_u32 s8
// DISASM-NEXT: s_add_co_ci_u32 s9
// DISASM-NEXT: s_set_pc_i64 s[8:9]

// METADATA: .name:           entry_tramp_kernel
// METADATA: .sgpr_count:     10

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --entry-trampolines --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf

// COM: If the requested entry trampoline cannot allocate an aligned scratch
// COM: SGPR pair, the rewrite fails instead of returning a partial output.
// RUN: sed 's/.sgpr_count: 8/.sgpr_count: 105/' %s > %t.highsgpr.s
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib \
// RUN:   %t.highsgpr.s -o %t.highsgpr.elf
// RUN: hotswap-rewrite %t.highsgpr.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --entry-trampolines --expect-status ERROR \
// RUN:   | %FileCheck --check-prefix=NO-SCRATCH %s
// NO-SCRATCH: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl entry_tramp_kernel
.p2align 8
.type entry_tramp_kernel,@function
entry_tramp_kernel:
  v_mov_b32_e32 v0, 0
  s_endpgm
.Lentry_tramp_kernel_end:
.size entry_tramp_kernel, .Lentry_tramp_kernel_end-entry_tramp_kernel

.rodata
.p2align 8
.amdhsa_kernel entry_tramp_kernel
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 1
  .amdhsa_inst_pref_size 7
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: entry_tramp_kernel
      .symbol: entry_tramp_kernel.kd
      .sgpr_count: 8
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
