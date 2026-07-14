// COM: Live tensor_load_to_lds descriptor SGPRs require save/restore around
// COM: the A0 D# Group 1 mask clear. If the kernel consumes all addressable
// COM: SGPRs, hotswap must fail instead of returning unsafe code.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefixes=LOG,API %s
// LOG: hotswap: error: tensor_load_to_lds descriptor save: no aligned block of 1 safe SGPRs fits below s106
// API: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_no_scratch
.p2align 8
.type test_tensor_no_scratch,@function
test_tensor_no_scratch:
  tensor_load_to_lds s[0:3], s[4:11]
  s_mov_b32 s0, s4
  s_endpgm
.Ltest_tensor_no_scratch_end:
.size test_tensor_no_scratch, .Ltest_tensor_no_scratch_end-test_tensor_no_scratch

.rodata
.p2align 8
.amdhsa_kernel test_tensor_no_scratch
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 106
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_tensor_no_scratch
      .symbol: test_tensor_no_scratch.kd
      .sgpr_count: 106
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
