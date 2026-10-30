# REQUIRES: x86
## Section properties that decide whether ICF folds two sections.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 fold.s -o fold.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 align.s -o align.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 differ.s -o differ.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 ineligible.s -o ineligible.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 data.s -o data.o

## Identical sections are folded, whether or not they are in a COMDAT group.
# RUN: ld.lld fold.o --icf=all --print-icf-sections | FileCheck %s --check-prefix=FOLD --implicit-check-not=section
# FOLD:      selected section fold.o:(.text.f1)
# FOLD-NEXT:   removing identical section fold.o:(.text.f2)

## The selected section takes the larger alignment.
# RUN: ld.lld align.o -o align --Ttext=0x220000 --icf=all --print-icf-sections | FileCheck %s --check-prefix=ALIGN --implicit-check-not=section
# RUN: llvm-objdump -t align | FileCheck %s --check-prefix=SYM
# ALIGN:      selected section align.o:(.text.f1)
# ALIGN-NEXT:   removing identical section align.o:(.text.f2)
# SYM: 0000000000220000 g .text 0000000000000000 _start
# SYM: 0000000000220100 g .text 0000000000000000 f1

## Sections that differ in contents, flags, or output section are not folded.
# RUN: ld.lld differ.o --icf=all --print-icf-sections | count 0

## Writable sections, .init, .fini, sections whose names are C identifiers,
## and SHF_LINK_ORDER sections are not folded.
# RUN: ld.lld ineligible.o --icf=all --print-icf-sections | count 0

## Without .addrsig, every symbol is address-significant, so read-only data
## sections defining symbols are folded only with
## --ignore-data-address-equality.
# RUN: ld.lld data.o --icf=all --print-icf-sections | count 0
# RUN: ld.lld data.o --icf=all --ignore-data-address-equality --print-icf-sections | FileCheck %s --check-prefix=DATA --implicit-check-not=section
# DATA:      selected section data.o:(.rodata.d1)
# DATA-NEXT:   removing identical section data.o:(.rodata.d2)
# DATA-NEXT: selected section data.o:(.data.rel.ro)
# DATA-NEXT:   removing identical section data.o:(.data.rel.ro.foo)

#--- fold.s
.globl _start
_start:
  ret

.section .text.f1,"ax"
  mov $60, %rax
  syscall

.section .text.f2,"axG",@progbits,foo,comdat
  mov $60, %rax
  syscall

#--- align.s
.globl _start, f1
_start:
  ret

.section .text.f1,"ax"
.p2align 0
f1:
  mov $60, %rax
  syscall

.section .text.f2,"ax"
.p2align 8
  mov $60, %rax
  syscall

#--- differ.s
.globl _start
_start:
  ret

.section .text.a1,"ax"
.byte 1
.section .text.a2,"ax"
.byte 2

.section .aaa,"ax",@progbits,unique,1
.byte 3
.section .aaa,"axS",@progbits,unique,2
.byte 3

.section .foo,"ax"
.byte 4
.section .bar,"ax"
.byte 4

#--- ineligible.s
.globl _start
_start:
  ret

.section .text.w1,"awx"
.byte 1
.section .text.w2,"awx"
.byte 1

.section .init,"ax",@progbits,unique,1
.byte 2
.section .init,"ax",@progbits,unique,2
.byte 2
.section .fini,"ax",@progbits,unique,1
.byte 3
.section .fini,"ax",@progbits,unique,2
.byte 3

.section foo,"ax",@progbits,unique,1
.byte 4
.section foo,"ax",@progbits,unique,2
.byte 4

.section .lo,"a",@progbits,unique,1
lo1:
.byte 5
.section .lo,"a",@progbits,unique,2
lo2:
.byte 6
.section .bar,"ao",@progbits,lo1,unique,1
.byte 7
.section .bar,"ao",@progbits,lo2,unique,2
.byte 7

#--- data.s
.globl _start, d1, d2
_start:
  ret

.section .rodata.d1,"a"
d1:
.byte 1
.section .rodata.d2,"a"
d2:
.byte 1

.section .data.rel.ro,"aw"
.quad foo
.section .data.rel.ro.foo,"aw"
foo:
.quad foo
