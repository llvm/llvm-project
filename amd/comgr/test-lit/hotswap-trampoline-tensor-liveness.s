// COM: Test isSgprLiveAfter edge cases for tensor_load_to_lds patching.
// COM: A branch instruction between the tensor_load and the next use of
// COM: the descriptor SGPR forces the heuristic to conservatively assume
// COM: the SGPR is live, producing save/restore even though the use may
// COM: not execute on all paths.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --strict-mode --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: Kernel 1 (branch guard): s_cbranch_scc1 sits between tensor_load
// COM: and s_mov (which reads s4). isSgprLiveAfter returns true at the
// COM: branch, so save/restore is emitted conservatively.
// DISASM-LABEL: <test_tensor_branch_guard>:
// DISASM: s_branch
// DISASM: s_cbranch_scc1
// DISASM: s_endpgm
// DISASM: s_mov_b32 [[SCRATCH:s[0-9]+]], s4
// DISASM-NEXT: s_pack_hh_b32_b16 s4, 0, s4
// DISASM-NEXT: tensor_load_to_lds
// DISASM-NEXT: s_mov_b32 s4, [[SCRATCH]]
// DISASM-NEXT: s_branch

// COM: Idempotency
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --strict-mode --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_branch_guard
.p2align 8
.type test_tensor_branch_guard,@function
test_tensor_branch_guard:
  tensor_load_to_lds s[0:3], s[4:11]
  s_cbranch_scc1 .Lskip
  s_mov_b32 s0, s4
.Lskip:
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
.Ltest_tensor_branch_guard_end:
.size test_tensor_branch_guard, .Ltest_tensor_branch_guard_end-test_tensor_branch_guard

.rodata
.p2align 8
.amdhsa_kernel test_tensor_branch_guard
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_tensor_branch_guard
      .symbol: test_tensor_branch_guard.kd
      .sgpr_count: 12
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
