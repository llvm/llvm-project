// COM: Test the s_add_pc_i64 long-branch trampoline path. When the appended
// COM: trampoline pool sits farther than s_branch's +-128 KB reach from the
// COM: patch site, both trampoline edges use s_add_pc_i64 (a PC-relative long
// COM: branch that reaches anywhere, needs no scratch register, and does not
// COM: touch SCC) instead of s_branch. A large .rept filler (~160 KB, non-NOP
// COM: so it forms no usable sled) pushes the pool past s_branch's real reach
// COM: from the tensor_load_to_lds at the top of the kernel.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: Forward edge: the site is overwritten in place with an 8-byte
// COM: s_add_pc_i64 long branch (the 12-byte tensor slot leaves room, padded
// COM: with s_nop), NOT an s_branch.
// DISASM-LABEL: <test_far>:
// DISASM: s_add_pc_i64
// DISASM-NEXT: s_nop
// DISASM-NEXT: s_endpgm

// COM: Trampoline body (appended after .text): the relocated multicast
// COM: mask-clear + tensor_load, then an s_add_pc_i64 long branch-back.
// DISASM: s_pack_hh_b32_b16 s4, 0, s4
// DISASM-NEXT: tensor_load_to_lds
// DISASM-NEXT: s_add_pc_i64

// COM: Idempotency: rewriting the output again must be a no-op.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_far
.p2align 8
.type test_far,@function
test_far:
  tensor_load_to_lds s[0:3], s[4:11]
  s_endpgm
  // ~160 KB of non-NOP filler so the appended trampoline pool is beyond
  // s_branch's +-128 KB reach from the tensor_load above (forces the
  // long-branch path).
  .rept 40000
    s_mov_b32 s0, s1
  .endr
.Ltest_far_end:
.size test_far, .Ltest_far_end-test_far

.rodata
.p2align 8
.amdhsa_kernel test_far
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel
