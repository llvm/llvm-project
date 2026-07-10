// COM: Test HotSwap strict-mode B0 tensor_load_to_lds scratch allocation when
// COM: the kernel descriptor under-reports SGPR use. Scratch must not be
// COM: allocated inside the tensor descriptor tuple s[4:11].

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   --strict-mode --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// DISASM-LABEL: <test_tensor_b0_descriptor_scratch_exclusion>:
// DISASM: s_branch
// DISASM: s_mov_b32 s0, s4
// DISASM: s_endpgm
// DISASM: s_mov_b32 s12, s4
// DISASM-NEXT: s_getreg_b32 s4, hwreg(HW_REG_IB_STS2, 6, 4)
// DISASM-NEXT: s_cmp_eq_u32 s4, 0
// DISASM-NEXT: s_pack_hh_b32_b16 s4, 0, s12
// DISASM-NEXT: s_cselect_b32 s4, s4, s12
// DISASM-NEXT: tensor_load_to_lds s[0:3], s[4:11]
// DISASM-NEXT: s_mov_b32 s4, s12
// DISASM-NEXT: s_branch

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_b0_descriptor_scratch_exclusion
.p2align 8
.type test_tensor_b0_descriptor_scratch_exclusion,@function
test_tensor_b0_descriptor_scratch_exclusion:
  tensor_load_to_lds s[0:3], s[4:11]
  s_mov_b32 s0, s4
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
.Ltest_tensor_b0_descriptor_scratch_exclusion_end:
.size test_tensor_b0_descriptor_scratch_exclusion, .Ltest_tensor_b0_descriptor_scratch_exclusion_end-test_tensor_b0_descriptor_scratch_exclusion

.rodata
.p2align 8
.amdhsa_kernel test_tensor_b0_descriptor_scratch_exclusion
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 4
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_tensor_b0_descriptor_scratch_exclusion
      .symbol: test_tensor_b0_descriptor_scratch_exclusion.kd
      .sgpr_count: 4
      .vgpr_count: 4
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
