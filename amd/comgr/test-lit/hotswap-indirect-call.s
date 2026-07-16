// COM: HSV-010 regression: a production PyTorch/AITER object contains the
// COM: register-target call `s_swap_pc_i64 s[30:31], s[0:1]`. The target is
// COM: not statically known, so direct-target collection must ignore it rather
// COM: than aborting an otherwise valid rewrite.
// COM:
// COM: Keep a real DS two-address patch in this reduced object so the rewrite
// COM: creates a trampoline and exercises the production failure path.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// DISASM-LABEL: <test_indirect_call>:
// DISASM: s_swap_pc_i64 s[30:31], s[0:1]
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NEXT: s_endpgm
// DISASM-NOT: ds_load_2addr_stride64_b32
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
.globl test_indirect_call
.p2align 8
.type test_indirect_call,@function
test_indirect_call:
  s_swap_pc_i64 s[30:31], s[0:1]
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
  s_wait_dscnt 0x0
  s_endpgm
.Ltest_indirect_call_end:
.size test_indirect_call, .Ltest_indirect_call_end-test_indirect_call

.rodata
.p2align 8
.amdhsa_kernel test_indirect_call
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 32
  .amdhsa_inst_pref_size 2
.end_amdhsa_kernel
