// COM: Test HotSwap strict-mode B0 tensor_load_to_lds masking. B0 must clear
// COM: D# Group 1 wg_mask bits [15:0] only when IB_STS2.CLUSTER_ID == 0.
// COM: Cluster loads do not need the A0 M0 workaround on a B0 target.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   --output %t.default.elf \
// RUN:   | %FileCheck --check-prefix=DEFAULTAPI %s
// DEFAULTAPI: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.default.elf | %FileCheck --check-prefix=DEFAULTDIS %s
// DEFAULTDIS-LABEL: <test_tensor_b0_dynamic>:
// DEFAULTDIS-NOT: s_getreg_b32
// DEFAULTDIS-NOT: s_pack_hh_b32_b16
// DEFAULTDIS: tensor_load_to_lds s[0:3], s[4:11]
// DEFAULTDIS: cluster_load_b32 v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9:]+}}]
// DEFAULTDIS: s_endpgm

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   --strict-mode --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// RUN: %llvm-readelf --notes %t.out.elf | %FileCheck --check-prefix=METADATA %s

// DISASM-LABEL: <test_tensor_b0_dynamic>:
// DISASM: s_branch
// DISASM: s_mov_b32 s0, s4
// DISASM-NEXT: cluster_load_b32 v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9:]+}}]
// DISASM-NOT: s_and_b32 m0
// DISASM: s_endpgm
// DISASM: s_mov_b32 [[SCR:s[0-9]+]], s4
// DISASM-NEXT: s_getreg_b32 s4, hwreg(HW_REG_IB_STS2, 6, 4)
// DISASM-NEXT: s_cmp_eq_u32 s4, 0
// DISASM-NEXT: s_pack_hh_b32_b16 s4, 0, [[SCR]]
// DISASM-NEXT: s_cselect_b32 s4, s4, [[SCR]]
// DISASM-NEXT: tensor_load_to_lds s[0:3], s[4:11]
// DISASM-NEXT: s_mov_b32 s4, [[SCR]]
// DISASM-NEXT: s_branch

// METADATA: .name:           test_tensor_b0_dynamic
// METADATA: .sgpr_count:     17

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   --strict-mode --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

// RUN: sed -e 's/.amdhsa_next_free_sgpr 16/.amdhsa_next_free_sgpr 106/' \
// RUN:     -e 's/.sgpr_count: 16/.sgpr_count: 106/' %s > %t.highsgpr.s
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib \
// RUN:   %t.highsgpr.s -o %t.highsgpr.elf
// RUN: hotswap-rewrite %t.highsgpr.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   --strict-mode --expect-status ERROR \
// RUN:   | %FileCheck --check-prefix=NO-SCRATCH %s
// NO-SCRATCH: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_b0_dynamic
.p2align 8
.type test_tensor_b0_dynamic,@function
test_tensor_b0_dynamic:
  tensor_load_to_lds s[0:3], s[4:11]
  s_mov_b32 s0, s4
  cluster_load_b32 v4, v1, s[6:7]
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
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
.Ltest_tensor_b0_dynamic_end:
.size test_tensor_b0_dynamic, .Ltest_tensor_b0_dynamic_end-test_tensor_b0_dynamic

.rodata
.p2align 8
.amdhsa_kernel test_tensor_b0_dynamic
  .amdhsa_next_free_vgpr 8
  .amdhsa_next_free_sgpr 16
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_tensor_b0_dynamic
      .symbol: test_tensor_b0_dynamic.kd
      .sgpr_count: 16
      .vgpr_count: 8
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
