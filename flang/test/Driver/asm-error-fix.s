! Test that flang rejects assembly files as input

! RUN: not %flang -c %s 2>&1 | FileCheck %s

! CHECK: error: flang does not accept assembly code

.globl foo
foo:
  ret
