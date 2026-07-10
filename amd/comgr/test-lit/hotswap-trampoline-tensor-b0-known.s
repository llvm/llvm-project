// COM: Test HotSwap B0 tensor_load_to_lds masking when fixed .cluster_dims
// COM: metadata proves the dispatch is non-cluster or size-one.

// RUN: %llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 \
// RUN:   --amdhsa-code-object-version=6 -filetype=obj %s -o %t.o
// RUN: %ld.lld -flavor gnu -m elf64_amdgpu %t.o -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   --output %t.default.elf \
// RUN:   | %FileCheck --check-prefix=DEFAULTAPI %s
// DEFAULTAPI: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.default.elf | %FileCheck --check-prefix=DEFAULTDIS %s
// DEFAULTDIS-LABEL: <test_tensor_b0_known_noncluster>:
// DEFAULTDIS-NOT: s_getreg_b32
// DEFAULTDIS-NOT: s_pack_hh_b32_b16
// DEFAULTDIS: tensor_load_to_lds s[0:3], s[4:11]
// DEFAULTDIS: s_endpgm

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   --strict-mode --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// RUN: %llvm-readelf --notes %t.out.elf | %FileCheck --check-prefix=METADATA %s

// DISASM-LABEL: <test_tensor_b0_known_noncluster>:
// DISASM: s_branch
// DISASM: s_endpgm
// DISASM-NOT: s_getreg_b32
// DISASM: s_pack_hh_b32_b16 s4, 0, s4
// DISASM-NEXT: tensor_load_to_lds s[0:3], s[4:11]
// DISASM-NEXT: s_branch
// DISASM-LABEL: <test_tensor_b0_known_cluster>:
// DISASM: s_branch
// DISASM: s_endpgm
// DISASM: s_mov_b32 [[SCR:s[0-9]+]], s4
// DISASM-NEXT: s_getreg_b32 s4, hwreg(HW_REG_IB_STS2, 6, 4)
// DISASM-NEXT: s_cmp_eq_u32 s4, 0
// DISASM-NEXT: s_pack_hh_b32_b16 s4, 0, [[SCR]]
// DISASM-NEXT: s_cselect_b32 s4, s4, [[SCR]]
// DISASM-NEXT: tensor_load_to_lds s[0:3], s[4:11]
// DISASM-NEXT: s_branch

// METADATA: .cluster_dims:
// METADATA-NEXT: - 1
// METADATA-NEXT: - 1
// METADATA-NEXT: - 1
// METADATA: .name:           test_tensor_b0_known_noncluster
// METADATA: .sgpr_count:     16
// METADATA: .cluster_dims:
// METADATA-NEXT: - 2
// METADATA-NEXT: - 1
// METADATA-NEXT: - 1
// METADATA: .name:           test_tensor_b0_known_cluster
// METADATA: .sgpr_count:     17

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   --strict-mode --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_b0_known_noncluster
.p2align 8
.type test_tensor_b0_known_noncluster,@function
test_tensor_b0_known_noncluster:
  tensor_load_to_lds s[0:3], s[4:11]
  s_endpgm
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
.Ltest_tensor_b0_known_noncluster_end:
.size test_tensor_b0_known_noncluster, .Ltest_tensor_b0_known_noncluster_end-test_tensor_b0_known_noncluster

.globl test_tensor_b0_known_cluster
.p2align 8
.type test_tensor_b0_known_cluster,@function
test_tensor_b0_known_cluster:
  tensor_load_to_lds s[0:3], s[4:11]
  s_endpgm
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
.Ltest_tensor_b0_known_cluster_end:
.size test_tensor_b0_known_cluster, .Ltest_tensor_b0_known_cluster_end-test_tensor_b0_known_cluster

.rodata
.p2align 8
.amdhsa_kernel test_tensor_b0_known_noncluster
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 16
.end_amdhsa_kernel

.amdhsa_kernel test_tensor_b0_known_cluster
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 16
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 6
    - 0
  amdhsa.kernels:
    - .name: test_tensor_b0_known_noncluster
      .symbol: test_tensor_b0_known_noncluster.kd
      .sgpr_count: 16
      .vgpr_count: 4
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
      .cluster_dims:
        - 1
        - 1
        - 1
    - .name: test_tensor_b0_known_cluster
      .symbol: test_tensor_b0_known_cluster.kd
      .sgpr_count: 16
      .vgpr_count: 4
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
      .cluster_dims:
        - 2
        - 1
        - 1
.end_amdgpu_metadata
