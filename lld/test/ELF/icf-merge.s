# REQUIRES: x86

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 kind.s -o kind.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 far.s -o far.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 end.s -o end.o

## f1 and f2 reach "foo" from different offsets in different files, and so do
## the section symbols g1 and g2. f3 reaches another string, f4 another offset.
# RUN: ld.lld a.o b.o -o ab --icf=all --print-icf-sections | \
# RUN:   FileCheck %s --implicit-check-not=section
# CHECK-DAG: selected section {{.*}}a.o:(.text.f1)
# CHECK-DAG:   removing identical section {{.*}}b.o:(.text.f2)
# CHECK-DAG: selected section {{.*}}a.o:(.text.g1)
# CHECK-DAG:   removing identical section {{.*}}b.o:(.text.g2)

## A reference into a merge section never equals one into a regular section.
# RUN: ld.lld kind.o -o kind --icf=all --print-icf-sections | count 0

## A section symbol's addend is not validated until getSymVA.
# RUN: not ld.lld far.o --icf=all 2>&1 | FileCheck %s --check-prefix=FAR --implicit-check-not=error:
# FAR: error: far.o:(.rodata.str1.1): offset 0xfffffffffffffffc is outside the section
# RUN: ld.lld end.o -o end --icf=all

#--- a.s
.globl _start
_start:
  ret

.section .rodata.str,"aMS",@progbits,1
foo:
.asciz "foo"
.asciz "string 1"

.section .text.f1,"ax"
lea foo+42(%rip), %rax

.section .text.g1,"ax"
.quad .rodata.str

#--- b.s
.section .rodata.str,"aMS",@progbits,1
.asciz "bar"
foo:
.asciz "foo"
boo:
.asciz "boo"

.section .text.f2,"ax"
lea foo+42(%rip), %rax

.section .text.f3,"ax"
lea boo+42(%rip), %rax

.section .text.f4,"ax"
lea foo+43(%rip), %rax

.section .text.g2,"ax"
.quad .rodata.str+4

#--- kind.s
.globl _start
_start:
  ret

.section .rodata.str,"aMS",@progbits,1
rodata:
.asciz "foo"

.section .text.foo,"ax"
.quad rodata

.section .text.bar,"ax"
.quad _start

#--- far.s
.globl _start
_start:
  ret

.section .rodata.str1.1,"aMS",@progbits,1
.asciz "aa"

.section .text.a,"ax"
leaq .rodata.str1.1(%rip), %rax

#--- end.s
.globl _start
_start:
  ret

.section .rodata.cst8,"aM",@progbits,8
.quad 0x1122334455667788

.section .text.a,"ax"
.quad .rodata.cst8+8
