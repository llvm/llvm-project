// COM: Remaining SGPR-relative cluster_load instructions use their SGPR tuple
// COM: through the wrapped instruction. If metadata under-reports SGPR usage,
// COM: scratch allocation must skip that tuple.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// RUN: %llvm-readelf --notes %t.out.elf | %FileCheck --check-prefix=METADATA %s

// DISASM-LABEL: <test_cluster_sgpr_scratch_exclusion>:
// DISASM: s_branch
// DISASM: s_endpgm
// DISASM: s_mov_b32 s6, m0
// DISASM-NEXT: s_pack_hh_b32_b16 m0, 0, m0
// DISASM-NEXT: cluster_load_b32 v{{[0-9]+}}, v{{[0-9]+}}, s[4:5]
// DISASM-NEXT: s_mov_b32 m0, s6

// METADATA: .name:           test_cluster_sgpr_scratch_exclusion
// METADATA: .sgpr_count:     7

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_cluster_sgpr_scratch_exclusion
.p2align 8
.type test_cluster_sgpr_scratch_exclusion,@function
test_cluster_sgpr_scratch_exclusion:
  cluster_load_b32 v4, v1, s[4:5]
  s_wait_loadcnt 0x0
  s_endpgm
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
.Ltest_cluster_sgpr_scratch_exclusion_end:
.size test_cluster_sgpr_scratch_exclusion, .Ltest_cluster_sgpr_scratch_exclusion_end-test_cluster_sgpr_scratch_exclusion

.rodata
.p2align 8
.amdhsa_kernel test_cluster_sgpr_scratch_exclusion
  .amdhsa_next_free_vgpr 6
  .amdhsa_next_free_sgpr 4
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_cluster_sgpr_scratch_exclusion
      .symbol: test_cluster_sgpr_scratch_exclusion.kd
      .sgpr_count: 4
      .vgpr_count: 6
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
