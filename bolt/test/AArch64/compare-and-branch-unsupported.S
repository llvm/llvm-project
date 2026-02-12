# This test checks that splitting functions which contain short range
# conditional branches does not work outside compact code model.

# REQUIRES: system-linux, asserts

# RUN: %clang %cflags -march=armv9-a+cmpbr -Wl,-q %s -o %t -DNUM_NOPS=0 -DRESERVE_SPACE=0
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: not llvm-bolt %t -o %t.bolt --data %t.fdata -split-functions 2>&1 \
# RUN:   | FileCheck %s

# RUN: %clang %cflags -march=armv9-a+cmpbr -Wl,-q %s -o %t -DNUM_NOPS=256 -DRESERVE_SPACE=0
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: not llvm-bolt %t -o %t.bolt --data %t.fdata -split-functions --keep-nops 2>&1 \
# RUN:   | FileCheck %s

# RUN: %clang %cflags -march=armv9-a+cmpbr -Wl,-q %s -o %t -DNUM_NOPS=0 -DRESERVE_SPACE=1
# RUN: link_fdata --no-lbr %s %t %t.fdata
# RUN: not llvm-bolt %t -o %t.bolt --data %t.fdata -split-functions 2>&1 \
# RUN:   | FileCheck %s

  .globl foo
  .type foo, %function
foo:
.entry_foo:
# FDATA: 1 foo #.entry_foo# 10
    cbgt x0, #0, .Lcold_foo
    mov x0, #1
.Lcold_foo:
    ret

  .globl large_function
  .type large_function, %function
large_function:
.entry_large_function:
# FDATA: 1 large_function #.entry_large_function# 10
  .rept NUM_NOPS
    nop
  .endr
    ret

.if RESERVE_SPACE
.space 0x8000000
.endif

## Force relocation mode.
.reloc 0, R_AARCH64_NONE

# CHECK: BOLT-ERROR: short range branch not supported outside compact code model
