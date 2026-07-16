// COM: A declared kernel immediately before a local return helper can reach
// COM: the helper by fallthrough without defining its link pair. The return
// COM: cannot be bounded even when its explicit caller is canonical.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefixes=LOG,API %s
// LOG: hotswap: s_set_pc_i64 at 0x8 is not a bounded return: declared entry at 0x0 falls through to function entry 0x4
// LOG: hotswap: unresolved call target
// LOG: hotswap: unresolved control-flow target disables NOP-sled emission,
// LOG-SAME: trampoline coalescing, source relocation, and .text gateways
// LOG: hotswap: error: no safe short-branch gateway for far site
// API: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.globl fallthrough_kernel
.protected fallthrough_kernel
.type fallthrough_kernel,@function
fallthrough_kernel:
  s_nop 0
.Lfallthrough_kernel_end:
.size fallthrough_kernel, .Lfallthrough_kernel_end-fallthrough_kernel

.type local_return_helper,@function
local_return_helper:
  s_nop 0
  s_set_pc_i64 s[0:1]
.Llocal_return_helper_end:
.size local_return_helper, .Llocal_return_helper_end-local_return_helper

.globl test_pc_materialized_fallthrough_entry
.p2align 8
.type test_pc_materialized_fallthrough_entry,@function
test_pc_materialized_fallthrough_entry:
  // Captured PC is .text+0x104 and the helper begins at .text+0x4.
  s_get_pc_i64 s[4:5]
  s_add_nc_u64 s[4:5], s[4:5], -256
  s_swap_pc_i64 s[0:1], s[4:5]
  ds_load_2addr_stride64_b64 v[0:3], v4 offset0:1 offset1:2
  s_wait_dscnt 0x0
  s_endpgm
.Ltest_pc_materialized_fallthrough_entry_end:
.size test_pc_materialized_fallthrough_entry, .Ltest_pc_materialized_fallthrough_entry_end-test_pc_materialized_fallthrough_entry

.fill 64, 1, 0
.rept 40000
  s_mov_b32 s0, s1
.endr

.rodata
.p2align 8
.amdhsa_kernel fallthrough_kernel
  .amdhsa_next_free_vgpr 0
  .amdhsa_next_free_sgpr 0
.end_amdhsa_kernel
.p2align 8
.amdhsa_kernel test_pc_materialized_fallthrough_entry
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 6
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: fallthrough_kernel
      .symbol: fallthrough_kernel.kd
      .sgpr_count: 0
      .vgpr_count: 0
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: test_pc_materialized_fallthrough_entry
      .symbol: test_pc_materialized_fallthrough_entry.kd
      .sgpr_count: 6
      .vgpr_count: 5
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
