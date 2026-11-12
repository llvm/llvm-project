# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o

## The definitions of symbol assignments may reference other symbols.
## Test we can handle them.

# RUN: echo "SECTIONS { aaa = foo | 1; .text  : { *(.text*) } }" > %t3.script
# RUN: ld.lld -o %t --script %t3.script %t.o
# RUN: llvm-nm -p %t | FileCheck --check-prefix=VAL1 %s

# VAL1: 0000000000000000 T foo
# VAL1: 0000000000000001 T aaa

# RUN: echo "SECTIONS { aaa = ABSOLUTE(foo - 1) + 1; .text  : { *(.text*) } }" > %t.script
# RUN: ld.lld -o %t --script %t.script %t.o
# RUN: llvm-nm -p %t | FileCheck --check-prefix=VAL %s

# RUN: echo "SECTIONS { aaa = 1 + ABSOLUTE(foo - 1); .text  : { *(.text*) } }" > %t.script
# RUN: ld.lld -o %t --script %t.script %t.o
# RUN: llvm-nm -p %t | FileCheck --check-prefix=VAL %s

# RUN: echo "SECTIONS { aaa = ABSOLUTE(foo); .text  : { *(.text*) } }" > %t4.script
# RUN: ld.lld -o %t --script %t4.script %t.o
# RUN: llvm-nm -p %t | FileCheck --check-prefix=VAL %s

# VAL: 0000000000000000 T foo
# VAL: 0000000000000000 A aaa

.section .text
.globl foo
foo:
