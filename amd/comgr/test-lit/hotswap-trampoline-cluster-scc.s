// COM: Test HotSwap A0 cluster_load M0 masking when SCC is live across the
// COM: cluster load. The M0 wg_mask clear must not clobber SCC before the
// COM: following s_cbranch_scc1.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// RUN: %llvm-readelf --notes %t.out.elf | %FileCheck --check-prefix=METADATA %s

// DISASM-LABEL: <test_cluster_scc_live>:
// DISASM: s_cmp_eq_u32 s0, s0
// DISASM: s_branch
// DISASM: s_cbranch_scc1
// DISASM: s_mov_b32 [[SCR:s[0-9]+]], m0
// DISASM-NEXT: s_pack_hh_b32_b16 m0, 0, m0
// DISASM-NEXT: cluster_load_b32 v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9:]+}}]
// DISASM-NEXT: s_mov_b32 m0, [[SCR]]
// DISASM-NEXT: s_branch
// DISASM-NOT: s_and_b32 m0

// METADATA: .name:           test_cluster_scc_live
// METADATA: .sgpr_count:     9

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_cluster_scc_live
.p2align 8
.type test_cluster_scc_live,@function
test_cluster_scc_live:
  s_cmp_eq_u32 s0, s0
  cluster_load_b32 v4, v1, s[2:3]
  s_cbranch_scc1 .Ldone
  s_mov_b32 s1, 0
.Ldone:
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
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
.Ltest_cluster_scc_live_end:
.size test_cluster_scc_live, .Ltest_cluster_scc_live_end-test_cluster_scc_live

.rodata
.p2align 8
.amdhsa_kernel test_cluster_scc_live
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 8
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_cluster_scc_live
      .symbol: test_cluster_scc_live.kd
      .sgpr_count: 8
      .vgpr_count: 5
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
