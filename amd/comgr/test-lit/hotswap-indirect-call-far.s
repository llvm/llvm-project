// COM: An unresolved register call may target any original text instruction.
// COM: In particular, it can target the second of two adjacent far patch sites.
// COM: The rewriter must not coalesce those sites or consume original .text
// COM: gateway space. This object has no safe target-independent route to the
// COM: far trampoline pool, so it must fail closed instead of returning a
// COM: successful object whose register call lands on rewritten padding.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefixes=LOG,API %s
// LOG: hotswap: unresolved control-flow target disables NOP-sled emission,
// LOG-SAME: trampoline coalescing, source relocation, and .text gateways
// LOG: hotswap: error: no safe short-branch gateway for far site
// API: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_indirect_call_far
.p2align 8
.type test_indirect_call_far,@function
test_indirect_call_far:
  // s_get_pc_i64 returns the address of the following instruction. The add
  // therefore makes s[2:3] point at the second DS instruction below.
  s_get_pc_i64 s[2:3]
  s_add_co_u32 s2, s2, 20
  s_add_co_ci_u32 s3, s3, 0
  s_swap_pc_i64 s[0:1], s[2:3]
  ds_load_2addr_stride64_b64 v[0:3], v4 offset0:1 offset1:2
.Lindirect_target:
  ds_load_2addr_stride64_b64 v[4:7], v8 offset0:3 offset1:4
  s_wait_dscnt 0x0
  s_endpgm
.size test_indirect_call_far, .-test_indirect_call_far

// Ordinarily this external zero-filled space can host long-branch gateways.
// An unresolved target could also land here, so the conservative path cannot
// consume it.
.fill 64, 1, 0

// Push the appended trampoline pool beyond s_branch's signed 16-bit dword
// range so success would require coalescing, relocation, or a .text gateway.
.rept 40000
  s_mov_b32 s0, s1
.endr

.rodata
.p2align 8
.amdhsa_kernel test_indirect_call_far
  .amdhsa_next_free_vgpr 9
  .amdhsa_next_free_sgpr 4
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_indirect_call_far
      .symbol: test_indirect_call_far.kd
      .sgpr_count: 4
      .vgpr_count: 9
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
