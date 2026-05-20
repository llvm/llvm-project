// COM: Test multi-DS stacking against a drain s_wait_dscnt: two
// COM: ds_load_2addr_stride64_b32 sites share a single s_wait_dscnt 0x0,
// COM: which must stay at 0x0 across both splits. The non-drain bump path
// COM: is covered by hotswap-trampoline-ds-pipelined.s.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: Both DS2 instructions are replaced by s_branch to their respective
// COM: expansion sleds; the shared drain wait stays at 0x0.
// DISASM-LABEL: <test_multi_ds>:
// DISASM-NOT: ds_load_2addr_stride64_b32
// DISASM: s_branch
// DISASM: s_branch
// DISASM: s_wait_dscnt 0x0

// COM: Idempotency
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_multi_ds
.p2align 8
.type test_multi_ds,@function
test_multi_ds:
  ds_load_2addr_stride64_b32 v[0:1], v4 offset0:0 offset1:1
  ds_load_2addr_stride64_b32 v[2:3], v4 offset0:2 offset1:3
  s_wait_dscnt 0x0
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
.Ltest_multi_ds_end:
.size test_multi_ds, .Ltest_multi_ds_end-test_multi_ds

.rodata
.p2align 8
.amdhsa_kernel test_multi_ds
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel
