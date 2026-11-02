# REQUIRES: x86
## Relocation properties that decide whether ICF folds two sections.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=i686 offset.s -o offset32.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-freebsd offset.s -o offset64.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 --crel offset.s -o offset-crel.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 differ.s -o differ.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 undef.s -o undef.o
# RUN: llvm-mc -filetype=obj -triple=i386 rel.s -o rel.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 abs.s -o abs.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 abs-same.s -o abs-same.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 abs-diff.s -o abs-diff.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 alias.s -o alias.o

## Sections whose relocations differ in offset are not folded.
# RUN: ld.lld offset32.o --icf=all --print-icf-sections | count 0
# RUN: ld.lld offset64.o --icf=all --print-icf-sections | count 0
# RUN: ld.lld offset-crel.o --icf=all --print-icf-sections | count 0

## Sections whose relocations differ in type, addend, or target are not folded.
# RUN: ld.lld differ.o --icf=all --print-icf-sections | count 0
# RUN: ld.lld undef.o -shared -z notext --icf=all --print-icf-sections | count 0

## REL implicit addends are compared as part of the contents.
# RUN: ld.lld rel.o --icf=all --print-icf-sections | FileCheck %s --check-prefix=REL --implicit-check-not=section
# REL:      selected section rel.o:(.text.f1)
# REL-NEXT:   removing identical section rel.o:(.text.f2)

## Relocations to absolute symbols are equal if the values are equal.
# RUN: ld.lld abs.o abs-same.o --icf=all --print-icf-sections | FileCheck %s --check-prefix=ABS --implicit-check-not=section
# RUN: ld.lld abs.o abs-diff.o --icf=all --print-icf-sections | count 0
# ABS:      selected section abs.o:(.text.f1)
# ABS-NEXT:   removing identical section abs.o:(.text.f2)

## Relocations to different symbols, e.g. aliases, in the same ineligible
## section are equal.
# RUN: ld.lld alias.o --keep-unique fu --icf=all --print-icf-sections | FileCheck %s --check-prefix=ALIAS --implicit-check-not=section
# ALIAS:      selected section alias.o:(.text.f1)
# ALIAS-NEXT:   removing identical section alias.o:(.text.f2)
# ALIAS-NEXT:   removing identical section alias.o:(.text.f3)
# ALIAS-NEXT: selected section alias.o:(.text.f4)
# ALIAS-NEXT:   removing identical section alias.o:(.text.f5)

#--- offset.s
.globl _start
_start:
  ret

.section .text.orig,"ax"
.quad -1

.section .text.foo,"ax"
.quad -1
.reloc 0, BFD_RELOC_NONE, 0

.section .text.bar,"ax"
.quad -1
.reloc 1, BFD_RELOC_NONE, 0

#--- differ.s
.globl _start, zed, zed2, e1, e2
_start:
  ret
zed:
  .byte 0
zed2:
  .byte 0

## Type.
.section .text.t1,"ax"
.quad 1
.reloc 0, R_X86_64_NONE, zed
.section .text.t2,"ax"
.quad 1
.reloc 0, R_X86_64_64, zed

## Addend.
.section .text.a1,"ax"
.quad _start+3
.section .text.a2,"ax"
.quad _start+4

## Targets with different values.
.section .text.v1,"ax"
.quad zed
.section .text.v2,"ax"
.quad zed2

## Targets in different ineligible sections.
.section .text.d1,"ax"
.quad d1
.section .text.d2,"ax"
.quad d2
.section .data.d1,"aw"
d1:
.quad 0
.section .data.d2,"aw"
d2:
.quad 0

## Targets in .eh_frame, which is not an InputSection.
.section .text.e1,"ax"
.quad e1
.section .text.e2,"ax"
.quad e2
.section .eh_frame,"a",@unwind
e1:
.quad 0
e2:
.quad 0

#--- undef.s
.globl _start, und
.hidden _start
_start:
  ret

.section .text.foo,"ax"
.quad _start
.section .text.bar,"ax"
.quad und

#--- rel.s
.globl _start
_start:
  ret

.section .text.f1,"ax"
.long _start+4
.section .text.f2,"ax"
.long _start+4
.section .text.f3,"ax"
.long _start+8

#--- abs.s
.globl _start
_start:
  ret

.section .text.f1,"ax"
.byte a1
.section .text.f2,"ax"
.byte a2

#--- abs-same.s
.globl a1, a2
a1 = 1
a2 = 1

#--- abs-diff.s
.globl a1, a2
a1 = 1
a2 = 2

#--- alias.s
.globl _start, d, d_alias, fu
_start:
  ret

.section .data.d,"aw"
d:
d_alias:
.long 42

.section .text.fu,"ax"
fu:
  nop

.section .text.f1,"ax"
.quad d
.section .text.f2,"ax"
.quad d_alias
.section .text.f3,"ax"
.quad .data.d
.section .text.f4,"ax"
.quad fu
.section .text.f5,"ax"
.quad .text.fu
