# REQUIRES: ppc
# RUN: llvm-mc -filetype=obj -triple=powerpc %s -o %t.o
# RUN: ld.lld %t.o %t.o -o /dev/null
# RUN: ld.lld -r --fatal-warnings %t.o %t.o -o /dev/null

## Similar to PPC64, clang/gcc PPC32 may emit a .rela.got2 which references a local symbol
## defined in a discarded .rodata section. Unfortunately, .got2 cannot be placed in a comdat
## because for lwz 3, .LC0-.LTOC(30), we cannot define .LC0 in a different .got2 section.

## Don't error "relocation refers to a discarded section".

.section .text.foo,"axG",@progbits,foo,comdat
.globl foo
foo:
 lwz 3, .LC0-.LTOC(30)
.L0:

.section .got2,"aw",@progbits
.set .LTOC, .got2+32768
.LC0:
.long .L0
