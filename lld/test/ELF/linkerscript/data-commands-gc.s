# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: echo "SECTIONS { .text : { *(.text*) QUAD(bar) } }" > %t.script
# RUN: ld.lld --gc-sections -o %t %t.o --script %t.script | FileCheck -allow-empty %s

# CHECK-NOT: unable to evaluate expression: input section .rodata.bar has no output section assigned

.section .rodata.bar
.quad 0x1122334455667788
.global bar
bar:

.section .text
.global _start
_start:
  nop
