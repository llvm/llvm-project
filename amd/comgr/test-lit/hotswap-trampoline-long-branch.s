// COM: HSV-009 / PLAT-205406: on gfx1250 A0 the far trampoline's backward
// COM: s_add_pc_i64 branch-back corrupts wave state (a GPU memory fault at
// COM: runtime). Until a scratch-register-based long branch-back lands, a patch
// COM: site beyond s_branch's +-128 KB reach of the appended pool is DECLINED
// COM: (left unpatched) instead of redirected through the long branch: the
// COM: original tensor_load_to_lds stays at the site and no s_add_pc_i64 (nor a
// COM: relocated trampoline body) is emitted. Near sites and in-place patches
// COM: are unaffected. A large .rept filler (~160 KB, non-NOP so it forms no
// COM: usable sled) pushes the pool past s_branch's reach to force the far case.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: Declined: the site keeps its original tensor_load_to_lds (it is NOT
// COM: overwritten with an s_add_pc_i64 forward branch), and no long-branch
// COM: redirect or relocated trampoline body (s_pack_hh_b32_b16 mask-clear) is
// COM: emitted anywhere.
// DISASM-LABEL: <test_far>:
// DISASM-NEXT: tensor_load_to_lds
// DISASM-NOT: s_add_pc_i64
// DISASM-NOT: s_pack_hh_b32_b16

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
