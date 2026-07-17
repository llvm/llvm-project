// COM: HotSwap applies the entry workaround when the entry-trampoline flag is
// COM: enabled. This multi-kernel layout uses aligned appended stubs because a
// COM: direct prefix would misalign later kernel entries.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.default.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// RUN: cmp %t.elf %t.default.elf
// RUN: %llvm-objdump -d %t.default.elf | %FileCheck --check-prefix=NO-TRAMP %s
// NO-TRAMP-LABEL: <entry_tramp_kernel>:
// NO-TRAMP-NEXT: v_mov_b32_e32 v0, 0
// NO-TRAMP-NEXT: s_endpgm
// NO-TRAMP-LABEL: <hipblaslt_entry_kernel>:
// NO-TRAMP: s_setreg_imm32_b32
// NO-TRAMP-LABEL: <decoder_trip_kernel>:
// NO-TRAMP: .long 0xffffffff
// NO-TRAMP-NOT: global_wb

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --entry-trampolines --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// RUN: %llvm-readelf --notes %t.out.elf | %FileCheck --check-prefix=METADATA %s

// COM: Entry trampolines are independent of the B0-to-A0 patch policy, so an
// COM: explicit B0->A0 rewrite should still redirect the descriptor to a stub.
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --entry-trampolines --output %t.b0a0.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// RUN: %llvm-objdump -d %t.b0a0.elf | %FileCheck --check-prefix=DISASM %s

// DISASM-LABEL: <entry_tramp_kernel>:
// DISASM-NEXT: v_mov_b32_e32 v0, 0
// DISASM-NEXT: s_endpgm
// DISASM-LABEL: <hipblaslt_entry_kernel>:
// DISASM-NEXT: s_setreg_imm32_b32
// DISASM-LABEL: <decoder_trip_kernel>:
// DISASM-NEXT: .long 0xffffffff
// DISASM: global_wb
// DISASM-NEXT: v_nop
// DISASM-NEXT: s_get_pc_i64
// DISASM: global_wb
// DISASM-NEXT: v_nop
// DISASM-NEXT: s_get_pc_i64
// DISASM: global_wb
// DISASM-NEXT: v_nop
// DISASM-NEXT: s_get_pc_i64

// METADATA: .name:           entry_tramp_kernel
// METADATA: .sgpr_count:     10

// COM: The alignment fallback records each appended stub for debuggers.
// RUN: %llvm-readelf -s %t.out.elf | %FileCheck --check-prefix=SYMS %s
// SYMS-DAG: entry_tramp_kernel.stub
// SYMS-DAG: hipblaslt_entry_kernel.stub
// SYMS-DAG: decoder_trip_kernel.stub

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --entry-trampolines --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf

// COM: The alignment fallback needs a scratch SGPR pair for its appended stubs.
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

.globl hipblaslt_entry_kernel
.p2align 8
.type hipblaslt_entry_kernel,@function
hipblaslt_entry_kernel:
  // Reduced from the gfx1250 hipBLASLt MXF8/BF16 smoke kernel entry. This is
  // real original kernel code, not an appended hotswap entry stub.
  s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2
  s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2
  s_and_b32 s63, 0x3fffffff, s2
  s_lshr_b32 s64, s2, 30
  s_mov_b32 s65, s3
  s_cmp_eq_u32 s64, 3
  s_cbranch_scc1 .Lhipblaslt_entry_done
  s_cmp_eq_u32 s64, 0
  s_cbranch_scc0 .Lhipblaslt_entry_done
.Lhipblaslt_entry_done:
  s_endpgm
.Lhipblaslt_entry_kernel_end:
.size hipblaslt_entry_kernel, .Lhipblaslt_entry_kernel_end-hipblaslt_entry_kernel

.globl decoder_trip_kernel
.p2align 8
.type decoder_trip_kernel,@function
decoder_trip_kernel:
  .long 0xffffffff
  s_endpgm
.Ldecoder_trip_kernel_end:
.size decoder_trip_kernel, .Ldecoder_trip_kernel_end-decoder_trip_kernel

.rodata
.p2align 8
.amdhsa_kernel entry_tramp_kernel
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 1
  .amdhsa_inst_pref_size 7
.end_amdhsa_kernel

.amdhsa_kernel hipblaslt_entry_kernel
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 66
.end_amdhsa_kernel

.amdhsa_kernel decoder_trip_kernel
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 1
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
    - .name: hipblaslt_entry_kernel
      .symbol: hipblaslt_entry_kernel.kd
      .sgpr_count: 66
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: decoder_trip_kernel
      .symbol: decoder_trip_kernel.kd
      .sgpr_count: 8
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
