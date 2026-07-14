// COM: A0 descriptor multicast bits stay clear after normalization. Even a
// COM: kernel declaring all user-addressable SGPRs therefore needs no scratch
// COM: register and must rewrite successfully without changing SGPR metadata.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// RUN: %llvm-readelf --notes %t.out.elf | %FileCheck --check-prefix=METADATA %s

// DISASM-LABEL: <test_tensor_no_scratch>:
// DISASM: s_branch
// DISASM: s_pack_hh_b32_b16 s4, 0, s4
// DISASM-NEXT: tensor_load_to_lds s[0:3], s[4:11]
// DISASM-NEXT: s_branch

// METADATA: .name:           test_tensor_no_scratch
// METADATA: .sgpr_count:     106

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
