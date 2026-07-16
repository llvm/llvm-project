// COM: Every known call entering a local return function must participate in
// COM: the link-register proof. A register-materialized call into the
// COM: function interior bypasses the entry and must keep the rewrite
// COM: fail-closed, even when another canonical call targets the entry.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefixes=LOG,API %s
// LOG: hotswap: s_set_pc_i64 at 0x4 is not a bounded return: call at 0x{{[0-9A-F]+}} enters the function interior at 0x4
// LOG: hotswap: unresolved call target
// LOG: hotswap: unresolved control-flow target disables NOP-sled emission,
// LOG-SAME: trampoline coalescing, source relocation, and .text gateways
// LOG: hotswap: error: no safe short-branch gateway for far site
// API: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.type local_return_helper,@function
local_return_helper:
  s_nop 0
.Llocal_return_epilogue:
  s_set_pc_i64 s[0:1]
.Llocal_return_helper_end:
.size local_return_helper, .Llocal_return_helper_end-local_return_helper

.globl test_pc_materialized_interior_entry
.p2align 8
.type test_pc_materialized_interior_entry,@function
test_pc_materialized_interior_entry:
  // Canonical entry call: captured PC is .text+0x104.
  s_get_pc_i64 s[4:5]
  s_add_nc_u64 s[4:5], s[4:5], -260
  s_swap_pc_i64 s[0:1], s[4:5]

  // Interior call: captured PC is .text+0x118 and the target is .text+0x4.
  // It also uses a different link pair from the helper return.
  s_get_pc_i64 s[6:7]
  s_add_nc_u64 s[6:7], s[6:7], -276
  s_swap_pc_i64 s[2:3], s[6:7]

  ds_load_2addr_stride64_b64 v[0:3], v4 offset0:1 offset1:2
  s_wait_dscnt 0x0
  s_endpgm
.Ltest_pc_materialized_interior_entry_end:
.size test_pc_materialized_interior_entry, .Ltest_pc_materialized_interior_entry_end-test_pc_materialized_interior_entry

.fill 64, 1, 0
.rept 40000
  s_mov_b32 s0, s1
.endr

.rodata
.p2align 8
.amdhsa_kernel test_pc_materialized_interior_entry
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 8
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_pc_materialized_interior_entry
      .symbol: test_pc_materialized_interior_entry.kd
      .sgpr_count: 8
      .vgpr_count: 5
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
