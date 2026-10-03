## A local conditional branch can share a block with a following external
## branch. Branch fixup must not add a redundant local branch after that exit.

# REQUIRES: system-linux

# RUN: %clang %cflags -Wl,-q -Wl,-e,foo %s -o %t -nostdlib
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: llvm-bolt %t -o %t.bolt --data %t.fdata --lite=0 --relax-exp \
# RUN:   --split-functions --split-strategy=all --print-after-branch-fixup \
# RUN:   --print-only=foo | FileCheck %s

# CHECK-LABEL: Binary Function "foo" after fix-branches {
# CHECK:      IsSplit     : 1
# CHECK:      b.eq
# CHECK-NEXT: {{.*}}b{{[[:space:]]}}
# CHECK-NEXT: Successors:
# CHECK: BOLT-INFO: starting experimental relaxation pass

  .text
  .globl bar
  .type bar, %function
bar:
  b .Lbar_ret
  .globl bar_body
  .type bar_body, %notype
bar_body:
  ret
.Lbar_ret:
  ret
  .size bar, .-bar

  .globl foo
  .type foo, %function
foo:
# FDATA: 1 foo #foo# 100
  cmp x0, #0
  b.eq .Lfoo_ret
  b bar_body
.Lfoo_ret:
  ret
  .size foo, .-foo

## Force relocation mode.
  .reloc 0, R_AARCH64_NONE
