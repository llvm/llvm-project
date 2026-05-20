// COM: Test HotSwap trampoline patch: non-drain s_wait_dscnt bump path.
// COM: Inputs use s_wait_dscnt 0x1 (pipelined wait permitting one in-flight
// COM: DS op), so each DS 2-addr split must increment the imm by 1. Inverse
// COM: of hotswap-trampoline-ds.s, which exercises drain preservation.
// COM:
// COM:   Kernel 1: one DS2 split  + s_wait_dscnt 0x1 -> bumped to 0x2.
// COM:   Kernel 2: two DS2 splits + s_wait_dscnt 0x1 -> bumped to 0x3.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: Kernel 1 (single split, +1 bump): one DS2 -> s_branch and the wait
// COM: incremented from 0x1 to 0x2.
// DISASM-LABEL: <test_ds_pipelined_single>:
// DISASM-NOT: ds_load_2addr_stride64_b32
// DISASM: s_branch
// DISASM: s_wait_dscnt 0x2
// DISASM: ds_load_b32 v0
// DISASM: ds_load_b32 v1
// DISASM: s_branch

// COM: Kernel 2 (two splits, +2 bump): two DS2 sites share one wait, which
// COM: is bumped twice from 0x1 to 0x3.
// DISASM-LABEL: <test_ds_pipelined_multi>:
// DISASM-NOT: ds_load_2addr_stride64_b32
// DISASM: s_branch
// DISASM: s_branch
// DISASM: s_wait_dscnt 0x3

// COM: Idempotency: a second rewrite must not bump 0x2 / 0x3 further.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

// ---- Kernel 1: single split, +1 bump (0x1 -> 0x2) ---------------------------

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_ds_pipelined_single
.p2align 8
.type test_ds_pipelined_single,@function
test_ds_pipelined_single:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
  s_wait_dscnt 0x1
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
.Ltest_ds_pipelined_single_end:
.size test_ds_pipelined_single, .Ltest_ds_pipelined_single_end-test_ds_pipelined_single

// ---- Kernel 2: two splits, +2 bump (0x1 -> 0x3) -----------------------------

.globl test_ds_pipelined_multi
.p2align 8
.type test_ds_pipelined_multi,@function
test_ds_pipelined_multi:
  ds_load_2addr_stride64_b32 v[0:1], v4 offset0:0 offset1:1
  ds_load_2addr_stride64_b32 v[2:3], v4 offset0:2 offset1:3
  s_wait_dscnt 0x1
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
.Ltest_ds_pipelined_multi_end:
.size test_ds_pipelined_multi, .Ltest_ds_pipelined_multi_end-test_ds_pipelined_multi

.rodata
.p2align 8
.amdhsa_kernel test_ds_pipelined_single
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdhsa_kernel test_ds_pipelined_multi
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel
