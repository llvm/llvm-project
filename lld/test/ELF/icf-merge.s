# REQUIRES: x86

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 same.s -o same.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 kind.s -o kind.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 parent.s -o parent.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 far.s -o far.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 far2.s -o far2.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 end.s -o end.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 end2.s -o end2.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 past.s -o past.o

## f1 and f2 reach "foo" from different offsets in different files, and so do
## the section symbols g1 and g2. f3 reaches another string, f4 another offset.
# RUN: ld.lld a.o b.o --icf=all --print-icf-sections | \
# RUN:   FileCheck %s --implicit-check-not=section
# CHECK-DAG: selected section {{.*}}a.o:(.text.f1)
# CHECK-DAG:   removing identical section {{.*}}b.o:(.text.f2)
# CHECK-DAG: selected section {{.*}}a.o:(.text.g1)
# CHECK-DAG:   removing identical section {{.*}}b.o:(.text.g2)

## References through the same section symbol with different addends are equal if
## they reach the same deduplicated piece, as are references through different symbols (f4).
# RUN: ld.lld same.o --icf=all --print-icf-sections | \
# RUN:   FileCheck %s --check-prefix=SAME --implicit-check-not=section
# SAME:      selected section same.o:(.text.f1)
# SAME-NEXT:   removing identical section same.o:(.text.f2)
# SAME-NEXT:   removing identical section same.o:(.text.f4)

## A reference into a merge section never equals one into a regular section.
# RUN: ld.lld kind.o --icf=all --print-icf-sections | count 0

## References into merge sections in different output sections are not equal.
# RUN: ld.lld parent.o --icf=all --print-icf-sections | count 0

## A section symbol's addend is not validated until getSymVA.
# RUN: not ld.lld far.o far2.o --icf=all 2>&1 | FileCheck %s --check-prefix=FAR --implicit-check-not=error:
# FAR: error: far.o:(.rodata.str1.1): offset 0xfffffffffffffffc is outside the section
# FAR: error: far2.o:(.rodata.str1.1): offset 0xfffffffffffffffc is outside the section
# RUN: not ld.lld end.o end2.o past.o --icf=all 2>&1 | FileCheck %s --check-prefix=END --implicit-check-not=error:
# END: error: past.o:(.rodata.cst8): offset 0x9 is outside the section

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

#--- same.s
.globl _start
_start:
  ret

.section .text.f1,"ax"
.quad .rodata.cst8

.section .text.f2,"ax"
.quad .rodata.cst8+8

.section .text.f3,"ax"
.quad .rodata.cst8+16

.section .text.f4,"ax"
.quad .rodata.cst8.1

.section .rodata.cst8,"aM",@progbits,8
.quad 42
.quad 42
.quad 43

.section .rodata.cst8.1,"aM",@progbits,8
.quad 42

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

#--- parent.s
.globl _start
_start:
  ret

.section .merge1,"aM",@progbits,8
.quad 42

.section .merge2,"aM",@progbits,8
.quad 42

.section .text.foo,"ax"
.quad .merge1

.section .text.bar,"ax"
.quad .merge2

#--- far.s
.globl _start
_start:
  ret

.section .rodata.str1.1,"aMS",@progbits,1
.asciz "aa"

.section .text.a,"ax"
leaq .rodata.str1.1(%rip), %rax

#--- far2.s
.section .rodata.str1.1,"aMS",@progbits,1
.asciz "bb"

.section .text.b,"ax"
leaq .rodata.str1.1(%rip), %rax

#--- end.s
.globl _start
_start:
  ret

.section .rodata.cst8,"aM",@progbits,8
.quad 0x1122334455667788

.section .text.a,"ax"
.quad .rodata.cst8+8

#--- end2.s
.section .rodata.cst8,"aM",@progbits,8
.quad 0x0112334455667789

.section .text.b,"ax"
.quad .rodata.cst8+8

#--- past.s
.section .rodata.cst8,"aM",@progbits,8
.quad 0

.section .text.c,"ax"
.quad .rodata.cst8+9
