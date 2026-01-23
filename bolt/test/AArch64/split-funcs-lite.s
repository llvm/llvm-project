# This test checks that tentative code layout for cold blocks always runs.
# It commonly happens when using lite mode with split functions.

# REQUIRES: system-linux, asserts

# RUN: %clang %cflags -o %t %s
# RUN: %clang %s %cflags -Wl,-q -o %t
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: llvm-bolt %t -o %t.bolt --data %t.fdata -split-functions \
# RUN:   -debug 2>&1 | FileCheck %s

  .text
  .globl  foo
  .type foo, %function
foo:
.entry_bb:
# FDATA: 1 foo #.entry_bb# 10
    cmp x0, #0
    b.eq .Lcold_bb1
    ret
.Lcold_bb1:
    ret

## Force relocation mode.
.reloc 0, R_AARCH64_NONE

# CHECK: foo{{.*}} cold tentative: {{.*}}
