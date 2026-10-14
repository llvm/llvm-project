! Test that flang rejects assembly files as input

! RUN: not %flang -c %s 2>&1 | FileCheck %s

! CHECK: error: flang does not support assembly files as input

.globl foo
foo:
  ret
