// COM: A declared function entry inside a PC-materialized call sequence can
// COM: bypass s_get_pc_i64 even when no direct branch targets that instruction.
// COM: The register call must remain unresolved, which disables every rewrite
// COM: that could consume an unknown original .text destination.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefixes=LOG,API %s
// LOG: hotswap: unresolved call target at 0x8 (s_swap_pc_i64)
// LOG: hotswap: unresolved control-flow target disables NOP-sled emission,
// LOG-SAME: trampoline coalescing, source relocation, and .text gateways
// LOG: hotswap: error: no safe short-branch gateway for far site
// API: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_pc_materialized_declared_entry
.p2align 8
.type test_pc_materialized_declared_entry,@function
test_pc_materialized_declared_entry:
  s_get_pc_i64 s[2:3]

// This symbol is an independent semantic entry. Entering here leaves s[2:3]
// caller-defined, so the apparent linear target calculation is not a proof.
.globl alternate_pc_materialization_entry
.type alternate_pc_materialization_entry,@function
alternate_pc_materialization_entry:
  s_add_nc_u64 s[2:3], s[2:3], 16
  s_swap_pc_i64 s[0:1], s[2:3]
.Lalternate_pc_materialization_entry_end:
.size alternate_pc_materialization_entry, .Lalternate_pc_materialization_entry_end-alternate_pc_materialization_entry

  ds_load_2addr_stride64_b64 v[0:3], v4 offset0:1 offset1:2
  ds_load_2addr_stride64_b64 v[4:7], v8 offset0:3 offset1:4
  s_wait_dscnt 0x0
  s_endpgm
.Ltest_pc_materialized_declared_entry_end:
.size test_pc_materialized_declared_entry, .Ltest_pc_materialized_declared_entry_end-test_pc_materialized_declared_entry

// This space would be usable for gateways only if every control-flow target
// were known.
.fill 64, 1, 0

// Keep the trampoline pool outside the signed s_branch range.
.rept 40000
  s_mov_b32 s0, s1
.endr

.rodata
.p2align 8
.amdhsa_kernel test_pc_materialized_declared_entry
  .amdhsa_next_free_vgpr 9
  .amdhsa_next_free_sgpr 4
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_pc_materialized_declared_entry
      .symbol: test_pc_materialized_declared_entry.kd
      .sgpr_count: 4
      .vgpr_count: 9
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
