# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %S/Inputs/passthru2.s -o %t2
# RUN: echo "SECTIONS { " \
# RUN:      "  /PASSTHRU/ : { *(.x*); *(.y*) }" \
# RUN:      "  .y.foo : { *(.x*); }" \
# RUN:      "  .x.foo : { *(.y*); }" \
# RUN:      "}" > %t.script
# RUN: ld.lld -o %t.out --script %t.script %t1 %t2
# RUN: llvm-objdump --section-headers %t.out | FileCheck %s

# The order should not be affected by the order of matches in the /PASSTHRU/
# section, and we should not have any .x.foo or .y.foo sections.

# CHECK: 1 .x.a
# CHECK: 2 .y.a
# CHECK: 3 .x.b
# CHECK: 4 .y.b

# CHECK-NOT: .x.foo
# CHECK-NOT: .y.foo

.section .x.a, "a"
  .quad 0

.section .y.a, "a"
  .quad 0
