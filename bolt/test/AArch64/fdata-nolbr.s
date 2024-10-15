# Check that using link_fdata tool in non-lbr mode allows using hardcoded
# addresses for basic block offsets.

# REQUIRES: system-linux

# RUN: %clang %cflags -o %t %s
# RUN: %clang %s %cflags -Wl,-q -o %t
# RUN: not link_fdata --no-lbr %s %t %t.fdata 2>&1 | FileCheck %s

  .text
  .globl  foo
  .type foo, %function
foo:
# FDATA: 1 foo 0 10
    ret

# Currently does not work on non-lbr mode.
# CHECK: AssertionError: ERROR: wrong format/whitespaces must be escaped
