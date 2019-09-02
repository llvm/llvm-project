# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o

## The definitions of symbol assignments may reference other symbols.
## Test we can handle them.

# RUN: echo "SECTIONS { aaa = foo | 1; .text  : { *(.text*) } }" > %t3.script
# RUN: ld.lld -o %t --script %t3.script %t.o
# RUN: llvm-objdump -t %t | FileCheck --check-prefix=VAL1 %s

# VAL1: 0000000000000000 .text 00000000 foo
# VAL1: 0000000000000001 .text 00000000 aaa

# RUN: echo "SECTIONS { aaa = ABSOLUTE(foo - 1) + 1; .text  : { *(.text*) } }" > %t.script
# RUN: ld.lld -o %t --script %t.script %t.o
# RUN: llvm-objdump -t %t | FileCheck --check-prefix=VAL %s

# RUN: echo "SECTIONS { aaa = 1 + ABSOLUTE(foo - 1); .text  : { *(.text*) } }" > %t.script
# RUN: ld.lld -o %t --script %t.script %t.o
# RUN: llvm-objdump -t %t | FileCheck --check-prefix=VAL %s

# RUN: echo "SECTIONS { aaa = ABSOLUTE(foo); .text  : { *(.text*) } }" > %t4.script
# RUN: ld.lld -o %t --script %t4.script %t.o
# RUN: llvm-objdump -t %t | FileCheck --check-prefix=VAL %s

# VAL: 0000000000000000 .text 00000000 foo
# VAL: 0000000000000000 *ABS* 00000000 aaa

.section .text
.globl foo
foo:
