# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: echo \
# RUN: "SECTIONS { . = 1000; .blah : { PROVIDE(foo = .); } }" \
# RUN:   > %t.script
# RUN: ld.lld -o %t1 --script %t.script %t -shared
# RUN: llvm-objdump -t %t1 | FileCheck %s
# CHECK: 00000000000003e8         *ABS*           00000000 foo

# RUN: echo \
# RUN: "SECTIONS { . = 1000; .blah : { PROVIDE_HIDDEN(foo = .); } }" \
# RUN:   > %t2.script
# RUN: ld.lld -o %t2 --script %t2.script %t -shared
# RUN: llvm-objdump -t %t2 | FileCheck %s --check-prefix=HIDDEN
# HIDDEN: 00000000000003e8         *ABS*           00000000 .hidden foo

.section blah
.globl patatino
patatino:
  movl $foo, %edx
