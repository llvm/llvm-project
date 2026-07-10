// COM: HSV-009 / PLAT-205406: on gfx1250 A0 the far trampoline's backward
// COM: s_add_pc_i64 branch-back corrupts wave state (a GPU memory fault at
// COM: runtime). Until a scratch-register-based long branch-back lands, a patch
// COM: site beyond s_branch's +-128 KB reach of the appended pool is DECLINED
// COM: instead of redirected through the long branch.
// COM:
// COM: The A0 tensor_load_to_lds mask workaround is required for correctness, so
// COM: hotswap must report ERROR if the trampoline patch cannot be emitted.
// COM: Optional far trampoline rewrites are still allowed to decline and return
// COM: success; hotswap-trampoline-ds-long-branch.s covers that behavior.
// COM: A large .rept filler (~160 KB, non-NOP so it forms no usable sled)
// COM: pushes the pool past s_branch's reach to force the far case.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: ERROR

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
