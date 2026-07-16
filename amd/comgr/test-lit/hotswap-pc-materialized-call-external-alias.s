// COM: A local return helper is not call-only when a GLOBAL PROTECTED kernel
// COM: alias and kernel descriptor expose the same entry. Kernel dispatch
// COM: does not define the helper's link pair, so the rewrite must fail closed.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefixes=LOG,API %s
// LOG: hotswap: s_set_pc_i64 at 0x4 is not a bounded return: externally reachable entry at 0x0 overlaps the local function
// LOG: hotswap: unresolved call target
// LOG: hotswap: unresolved control-flow target disables NOP-sled emission,
// LOG-SAME: trampoline coalescing, source relocation, and .text gateways
// LOG: hotswap: error: no safe short-branch gateway for far site
// API: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.type local_return_helper,@function
.globl aliased_return_kernel
.protected aliased_return_kernel
.type aliased_return_kernel,@function
local_return_helper:
aliased_return_kernel:
  s_nop 0
  s_set_pc_i64 s[0:1]
.Laliased_return_end:
.size local_return_helper, .Laliased_return_end-local_return_helper
.size aliased_return_kernel, .Laliased_return_end-aliased_return_kernel

.globl test_pc_materialized_external_alias
.p2align 8
.type test_pc_materialized_external_alias,@function
test_pc_materialized_external_alias:
  s_get_pc_i64 s[4:5]
  s_add_nc_u64 s[4:5], s[4:5], -260
  s_swap_pc_i64 s[0:1], s[4:5]
  ds_load_2addr_stride64_b64 v[0:3], v4 offset0:1 offset1:2
  s_wait_dscnt 0x0
  s_endpgm
.Ltest_pc_materialized_external_alias_end:
.size test_pc_materialized_external_alias, .Ltest_pc_materialized_external_alias_end-test_pc_materialized_external_alias

.fill 64, 1, 0
.rept 40000
  s_mov_b32 s0, s1
.endr

.rodata
.p2align 8
.amdhsa_kernel aliased_return_kernel
  .amdhsa_next_free_vgpr 0
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.p2align 8
.amdhsa_kernel test_pc_materialized_external_alias
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 6
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: aliased_return_kernel
      .symbol: aliased_return_kernel.kd
      .sgpr_count: 2
      .vgpr_count: 0
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: test_pc_materialized_external_alias
      .symbol: test_pc_materialized_external_alias.kd
      .sgpr_count: 6
      .vgpr_count: 5
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
