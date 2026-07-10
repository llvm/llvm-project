// COM: Test HotSwap strict-mode B0 tensor_load_to_lds masking when SCC is
// COM: live across the tensor load. The B0 runtime cluster-id guard must save
// COM: and restore SCC around its injected s_cmp_eq_u32.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   --strict-mode --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// RUN: %llvm-readelf --notes %t.out.elf | %FileCheck --check-prefix=METADATA %s

// DISASM-LABEL: <test_tensor_b0_scc_live>:
// DISASM: s_cmp_eq_u32 s0, s0
// DISASM-NEXT: s_branch
// DISASM: s_cbranch_scc1
// DISASM: s_mov_b32 s16, s4
// DISASM-NEXT: s_cselect_b32 s17, 1, 0
// DISASM-NEXT: s_getreg_b32 s4, hwreg(HW_REG_IB_STS2, 6, 4)
// DISASM-NEXT: s_cmp_eq_u32 s4, 0
// DISASM-NEXT: s_pack_hh_b32_b16 s4, 0, s16
// DISASM-NEXT: s_cselect_b32 s4, s4, s16
// DISASM-NEXT: s_cmp_lg_u32 s17, 0
// DISASM-NEXT: tensor_load_to_lds s[0:3], s[4:11]
// DISASM-NEXT: s_mov_b32 s4, s16
// DISASM-NEXT: s_branch

// METADATA: .name:           test_tensor_b0_scc_live
// METADATA: .sgpr_count:     18

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   --strict-mode --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_b0_scc_live
.p2align 8
.type test_tensor_b0_scc_live,@function
test_tensor_b0_scc_live:
  s_cmp_eq_u32 s0, s0
  tensor_load_to_lds s[0:3], s[4:11]
  s_cbranch_scc1 .Ldone
  s_mov_b32 s1, 0
.Ldone:
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
.Ltest_tensor_b0_scc_live_end:
.size test_tensor_b0_scc_live, .Ltest_tensor_b0_scc_live_end-test_tensor_b0_scc_live

.rodata
.p2align 8
.amdhsa_kernel test_tensor_b0_scc_live
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 16
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_tensor_b0_scc_live
      .symbol: test_tensor_b0_scc_live.kd
      .sgpr_count: 16
      .vgpr_count: 4
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
