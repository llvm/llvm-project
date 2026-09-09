## Check that a direct B to a function-body symbol is not relaxed as an
## ABI tail call. Such a branch can carry live caller-saved registers, so
## relaxation must use B-only branch chains instead of ADRP/BR call thunks.
##
## In the TAILCALL run, bar_body is at the function entry and the branch is an
## ABI tail call. In the BRANCH run, bar_body is inside the function body and
## must be relaxed as a branch.

# REQUIRES: system-linux

# RUN: %clang %cflags -Wl,-q -Wl,-e,foo %s -o %t.tailcall -nostdlib
# RUN: link_fdata --no-lbr %s %t.tailcall %t.tailcall.fdata
# RUN: llvm-bolt %t.tailcall -o %t.tailcall.bolt \
# RUN:   --data %t.tailcall.fdata --relocs \
# RUN:   --lite=0 --relax-exp --reorder-functions=exec-count \
# RUN:   --skip-funcs='^bar$' \
# RUN:   | FileCheck %s --check-prefix=TAILCALL-BOLT
# RUN: llvm-objdump -d \
# RUN:   --disassemble-symbols=foo,bar,bar_body,padding,__AArch64_forward_long_call_bar \
# RUN:   %t.tailcall.bolt | FileCheck %s --check-prefix=TAILCALL

# RUN: %clang %cflags -DBRANCH -Wl,-q -Wl,-e,foo %s -o %t.branch \
# RUN:   -nostdlib
# RUN: link_fdata --no-lbr %s %t.branch %t.branch.fdata
# RUN: llvm-bolt %t.branch -o %t.branch.bolt \
# RUN:   --data %t.branch.fdata --relocs \
# RUN:   --lite=0 --relax-exp --reorder-functions=exec-count \
# RUN:   | FileCheck %s --check-prefix=BRANCH-BOLT
# RUN: llvm-objdump -d \
# RUN:   --disassemble-symbols=foo,bar,bar_body,.bar_ret,padding,__AArch64_forward_branch_chain_0 \
# RUN:   %t.branch.bolt | FileCheck %s --check-prefix=BRANCH

# TAILCALL-BOLT: BOLT-INFO: relaxing branches for compact code model (<128MB)
# TAILCALL-BOLT: BOLT-INFO: starting experimental relaxation pass
# TAILCALL-BOLT: BOLT-INFO: relaxed 1 long cluster calls with thunks
# TAILCALL-BOLT: BOLT-INFO: 1 long thunks created

# BRANCH-BOLT: BOLT-INFO: relaxing branches for compact code model (<128MB)
# BRANCH-BOLT: BOLT-INFO: starting experimental relaxation pass
# BRANCH-BOLT: BOLT-INFO: relaxed 1 cross-cluster branches
# BRANCH-BOLT: BOLT-INFO: 1 branch thunks created

  .text
  .globl bar
  .type bar, %function
bar:
#ifdef BRANCH
.bar_entry:
  b .bar_ret
#endif
  .globl bar_body
  .type bar_body, %notype
bar_body:
  cmp x16, #42
  cset w0, ne
  ret
#ifdef BRANCH
.bar_ret:
  ret
#endif
  .size bar, .-bar

  .globl foo
  .type foo, %function
foo:
.foo_entry:
# FDATA: 1 foo #.foo_entry# 100
  mov x16, #42
  b bar_body
  .space 0x4400000
  .size foo, .-foo

  .globl padding
  .type padding, %function
padding:
.padding_entry:
# FDATA: 1 padding #.padding_entry# 100
  ret
  .space 0x4400000
  .size padding, .-padding

## Force relocation mode.
  .reloc 0, R_AARCH64_NONE

# TAILCALL: Disassembly of section .bolt.org.text:
#
# TAILCALL:      <bar>:
# TAILCALL-NEXT: <bar_body>:
# TAILCALL-NEXT:   {{.*}} cmp x16, #0x2a
# TAILCALL-NEXT:   {{.*}} cset w0, ne
# TAILCALL-NEXT:   {{.*}} ret
#
# TAILCALL: Disassembly of section .text:
#
# TAILCALL:      <foo>:
# TAILCALL-NEXT:   {{.*}} mov x16, #0x2a
# TAILCALL-NEXT:   {{.*}} b   {{.*}} <__AArch64_forward_long_call_bar>
#
# TAILCALL:      <__AArch64_forward_long_call_bar>:
# TAILCALL-NEXT:   {{.*}} adrp x16,
# TAILCALL-NEXT:   {{.*}} add x16, x16,
# TAILCALL-NEXT:   {{.*}} br x16
#
# TAILCALL:      <padding>:
# TAILCALL-NEXT:   {{.*}} ret


# BRANCH: Disassembly of section .bolt.org.text:
#
# BRANCH:      <bar_body>:
# BRANCH-NEXT:   {{.*}} cmp x16, #0x2a
# BRANCH-NEXT:   {{.*}} cset w0, ne
# BRANCH-NEXT:   {{.*}} ret
#
# BRANCH: Disassembly of section .text:
#
# BRANCH:      <foo>:
# BRANCH-NEXT:   {{.*}} mov x16, #0x2a
# BRANCH-NEXT:   {{.*}} b   0x[[CHAIN:[0-9a-f]+]] <__AArch64_forward_branch_chain_0>
#
# BRANCH:      <__AArch64_forward_branch_chain_0>:
# BRANCH-NEXT: [[CHAIN]]: {{.*}} b   0x[[BODY:[0-9a-f]+]] <bar+0x4>
#
# BRANCH:      <padding>:
# BRANCH-NEXT:   {{.*}} ret
#
# BRANCH: Disassembly of section .text.cold:
#
# BRANCH:      <bar>:
# BRANCH-NEXT:           {{.*}} b   0x[[RET:[0-9a-f]+]] <.bar_ret>
# BRANCH-NEXT: [[BODY]]: {{.*}} cmp x16, #0x2a
# BRANCH-NEXT:           {{.*}} cset w0, ne
# BRANCH-NEXT:           {{.*}} ret
#
# BRANCH:      <.bar_ret>:
# BRANCH-NEXT: [[RET]]: {{.*}} ret
