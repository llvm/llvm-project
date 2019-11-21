# REQUIRES: arm
## For --gc-sections, group members are retained or discarded as a unit.
## However, discarding a section via /DISCARD/ should not discard other members
## within the group. This is compatible with GNU ld.

# RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t.o

## We can discard .ARM.exidx* in a group.
# RUN: echo 'SECTIONS { /DISCARD/ : { *(.ARM.exidx*) }}' > %t.noarm.script
# RUN: ld.lld %t.o --gc-sections -T %t.noarm.script -o %t.noarm
# RUN: llvm-readobj -S %t.noarm | FileCheck %s --check-prefix=NOARM --implicit-check-not='Name: .ARM.exidx'

# NOARM: Name: .text
# NOARM: Name: .note._start

## Another example, we can discard SHT_NOTE in a group.
# RUN: echo 'SECTIONS { /DISCARD/ : { *(.note*) }}' > %t.nonote.script
# RUN: ld.lld %t.o --gc-sections -T %t.nonote.script -o %t.nonote
# RUN: llvm-readobj -S %t.nonote | FileCheck %s --check-prefix=NONOTE --implicit-check-not='Name: .note'

# NONOTE: Name: .ARM.exidx
# NONOTE: Name: .text

.section .text._start,"axG",%progbits,_start,comdat
.globl _start
_start:
.fnstart
.cantunwind
bx lr
.fnend

.section .note._start,"G",%note,_start,comdat
.byte 0
