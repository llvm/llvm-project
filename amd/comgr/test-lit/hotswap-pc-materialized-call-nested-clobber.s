// COM: A nested call can clobber its caller's return pair even when the call
// COM: instruction writes a different link pair. Without interprocedural
// COM: clobber proof, the outer s_set_pc_i64 must remain unresolved and the
// COM: rewrite must use the existing fail-closed path.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefixes=LOG,API %s
// LOG: hotswap: s_set_pc_i64 at 0x4 is not a bounded return:
// LOG-SAME: nested call at 0x0 may clobber the link register
// LOG: hotswap: unresolved call target
// LOG: hotswap: unresolved control-flow target disables NOP-sled emission,
// LOG-SAME: trampoline coalescing, source relocation, and .text gateways
// LOG: hotswap: error: no safe short-branch gateway for far site
// API: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

.type local_outer,@function
local_outer:
  s_call_i64 s[4:5], 1
  s_set_pc_i64 s[0:1]
.Llocal_outer_end:
.size local_outer, .Llocal_outer_end-local_outer

.type local_callee,@function
local_callee:
  s_mov_b32 s0, 0
  s_set_pc_i64 s[4:5]
.Llocal_callee_end:
.size local_callee, .Llocal_callee_end-local_callee

.globl test_nested_clobber
.p2align 8
.type test_nested_clobber,@function
test_nested_clobber:
  // Captured PC is .text+0x104 and local_outer starts at .text+0.
  s_get_pc_i64 s[6:7]
  s_add_nc_u64 s[6:7], s[6:7], -260
  s_swap_pc_i64 s[0:1], s[6:7]
  ds_load_2addr_stride64_b64 v[0:3], v4 offset0:1 offset1:2
  s_wait_dscnt 0x0
  s_endpgm
.Ltest_end:
.size test_nested_clobber, .Ltest_end-test_nested_clobber

.fill 64, 1, 0
.rept 40000
  s_mov_b32 s0, s1
.endr

.rodata
.p2align 8
.amdhsa_kernel test_nested_clobber
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 8
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_nested_clobber
      .symbol: test_nested_clobber.kd
      .sgpr_count: 8
      .vgpr_count: 5
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
