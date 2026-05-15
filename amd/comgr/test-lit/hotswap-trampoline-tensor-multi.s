// COM: Test multi-site tensor_load_to_lds patching: multiple tensor_load
// COM: instructions in a single kernel. Verifies:
// COM:   - Each site is independently patched with its own s_pack_hh
// COM:   - Idempotency guard correctly handles back-to-back patches
// COM:   - DS + tensor coexistence in the same kernel

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: Kernel 1: two tensor_load_to_lds with different descriptors
// COM: (s[4:11] and s[16:23]). Both should be patched independently.
// COM: The second tensor_load's predecessor after patching is the first's
// COM: branch-back, not its own s_pack_hh — idempotency guard must not
// COM: false-positive on it.
// DISASM-LABEL: <test_tensor_multi_different>:
// DISASM: s_branch
// DISASM: s_branch
// DISASM: s_endpgm
// DISASM: s_pack_hh_b32_b16
// DISASM: tensor_load_to_lds
// DISASM: s_branch
// DISASM: s_pack_hh_b32_b16
// DISASM: tensor_load_to_lds
// DISASM: s_branch

// COM: Kernel 2: two tensor_load_to_lds sharing the same descriptor
// COM: (s[4:11]). Both should still be patched — the idempotency guard
// COM: checks the immediately preceding instruction, and after patching
// COM: the first, the second's predecessor is an s_branch (not s_pack_hh).
// DISASM-LABEL: <test_tensor_multi_same>:
// DISASM: s_branch
// DISASM: s_branch
// DISASM: s_endpgm
// DISASM: s_pack_hh_b32_b16
// DISASM: tensor_load_to_lds
// DISASM: s_branch
// DISASM: s_pack_hh_b32_b16
// DISASM: tensor_load_to_lds
// DISASM: s_branch

// COM: Kernel 3: mixed DS 2-addr + tensor_load in the same kernel.
// COM: Both patch types should coexist: DS expansion produces two
// COM: single-address loads + wait bump, tensor produces s_pack_hh.
// DISASM-LABEL: <test_tensor_mixed_ds>:
// DISASM-NOT: ds_load_2addr_stride64_b32
// DISASM: s_branch
// DISASM: s_wait_dscnt
// DISASM: s_branch
// DISASM: s_endpgm

// COM: Idempotency
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

// ---- Kernel 1: two tensor_loads with different descriptors -----------------

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_tensor_multi_different
.p2align 8
.type test_tensor_multi_different,@function
test_tensor_multi_different:
  tensor_load_to_lds s[0:3], s[4:11]
  tensor_load_to_lds s[0:3], s[16:23]
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
.Ltest_tensor_multi_different_end:
.size test_tensor_multi_different, .Ltest_tensor_multi_different_end-test_tensor_multi_different

// ---- Kernel 2: two tensor_loads sharing the same descriptor ----------------

.globl test_tensor_multi_same
.p2align 8
.type test_tensor_multi_same,@function
test_tensor_multi_same:
  tensor_load_to_lds s[0:3], s[4:11]
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
.Ltest_tensor_multi_same_end:
.size test_tensor_multi_same, .Ltest_tensor_multi_same_end-test_tensor_multi_same

// ---- Kernel 3: mixed DS 2-addr + tensor_load in one kernel -----------------

.globl test_tensor_mixed_ds
.p2align 8
.type test_tensor_mixed_ds,@function
test_tensor_mixed_ds:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
  s_wait_dscnt 0x0
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
.Ltest_tensor_mixed_ds_end:
.size test_tensor_mixed_ds, .Ltest_tensor_mixed_ds_end-test_tensor_mixed_ds

.rodata
.p2align 8
.amdhsa_kernel test_tensor_multi_different
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 24
.end_amdhsa_kernel

.p2align 8
.amdhsa_kernel test_tensor_multi_same
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.p2align 8
.amdhsa_kernel test_tensor_mixed_ds
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel
