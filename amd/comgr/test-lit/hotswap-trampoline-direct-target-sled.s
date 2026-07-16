// COM: A direct branch target into a NOP run makes the run executable program
// COM: text, not scratch padding. collectDirectBranchTargets must therefore
// COM: run before patch emission so emitReplacementCode cannot place a
// COM: relocated replacement at the branch target.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: The direct branch still lands on NOP padding. The DS2 replacement is
// COM: emitted later in the appended trampoline pool, not at .Ldirect_target.
// DISASM-LABEL: <test_direct_target_sled>:
// DISASM-NOT: ds_load_2addr_stride64_b32
// DISASM: s_cbranch_scc1
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NEXT: s_endpgm
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_nop 0
// DISASM: ds_load_b32 v0
// DISASM-NEXT: ds_load_b32 v1
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NEXT: s_branch

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_direct_target_sled
.p2align 8
.type test_direct_target_sled,@function
test_direct_target_sled:
  s_cbranch_scc1 .Ldirect_target
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
  s_wait_dscnt 0x0
  s_endpgm
.Ldirect_target:
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_endpgm
.Ltest_direct_target_sled_end:
.size test_direct_target_sled, .Ltest_direct_target_sled_end-test_direct_target_sled

.rodata
.p2align 8
.amdhsa_kernel test_direct_target_sled
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 1
  .amdhsa_inst_pref_size 2
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_direct_target_sled
      .symbol: test_direct_target_sled.kd
      .sgpr_count: 1
      .vgpr_count: 3
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
      .uses_dynamic_stack: false
.end_amdgpu_metadata
