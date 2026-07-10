// COM: Explicit A0-to-A0 rewriting must not select the B0-to-A0 A0 mask
// COM: workarounds, even when the kernel contains mask-sensitive instructions.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific- \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: cmp %t.elf %t.out.elf

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_a0_to_a0_mask_noop
.p2align 8
.type test_a0_to_a0_mask_noop,@function
test_a0_to_a0_mask_noop:
  tensor_load_to_lds s[0:3], s[4:11]
  cluster_load_b32 v4, v1, s[12:13]
  s_wait_loadcnt 0x0
  s_endpgm
.Ltest_a0_to_a0_mask_noop_end:
.size test_a0_to_a0_mask_noop, .Ltest_a0_to_a0_mask_noop_end-test_a0_to_a0_mask_noop

.rodata
.p2align 8
.amdhsa_kernel test_a0_to_a0_mask_noop
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 14
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_a0_to_a0_mask_noop
      .symbol: test_a0_to_a0_mask_noop.kd
      .sgpr_count: 14
      .vgpr_count: 6
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
