# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: echo "SECTIONS { /DISCARD/ : { *(.eh_frame) } }" > %t.script
# RUN: ld.lld --eh-frame-hdr --script %t.script %t.o -o %t

.global _start
_start:
 nop

.section .foo,"ax",@progbits
.cfi_startproc
 nop
.cfi_endproc
