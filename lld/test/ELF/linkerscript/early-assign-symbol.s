# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o

# RUN: echo "SECTIONS { aaa = 1 + ABSOLUTE(foo - 1); .text  : { *(.text*) } }" > %t1.script
# RUN: not ld.lld -o %t --script %t1.script %t.o 2>&1 | FileCheck %s

# RUN: echo "SECTIONS { aaa = ABSOLUTE(foo - 1) + 1; .text  : { *(.text*) } }" > %t2.script
# RUN: not ld.lld -o %t --script %t2.script %t.o 2>&1 | FileCheck %s

# RUN: echo "SECTIONS { aaa = foo | 1; .text  : { *(.text*) } }" > %t3.script
# RUN: not ld.lld -o %t --script %t3.script %t.o 2>&1 | FileCheck %s

# CHECK: error: {{.*}}.script:1: unable to evaluate expression: input section .text has no output section assigned

# Simple case that we can handle.
# RUN: echo "SECTIONS { aaa = ABSOLUTE(foo); .text  : { *(.text*) } }" > %t4.script
# RUN: ld.lld -o %t --script %t4.script %t.o

.section .text
.globl foo
foo:
