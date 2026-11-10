# REQUIRES: aarch64
## Identical sections are only folded if their authenticated pointers have the
## same signing schema. __cfstring sections are always deduplicated, ignoring
## the bytes that relocations overwrite, so only the relocations can tell these
## apart; arm64e compilers sign each CFString's isa pointer.

# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t.o %s
# RUN: %lld -arch arm64e -platform_version macos 13.0 13.0 %t.o -o %t.out
# RUN: llvm-objdump --macho --syms %t.out | FileCheck %s
# RUN: %lld -arch arm64e -platform_version macos 13.0 13.0 --icf=all %t.o \
# RUN:   -o %t.icf.out
# RUN: llvm-objdump --macho --syms %t.icf.out | FileCheck %s

# CHECK:      [[#%x,SAME:]]   l O __DATA_CONST,__cfstring _same_a
# CHECK-NEXT: {{0*}}[[#SAME]]      l O __DATA_CONST,__cfstring _same_b
# CHECK-NEXT: {{0*}}[[#SAME+0x20]] l O __DATA_CONST,__cfstring _other_diversity
# CHECK-NEXT: {{0*}}[[#SAME+0x40]] l O __DATA_CONST,__cfstring _unsigned

.globl _main
.p2align 2
_main:
  ret

.section __DATA,__cfstring
.p2align 3
_same_a:
  .quad ___CFConstantStringClassReference@AUTH(da,27361,addr)
  .long 1992
  .space 4
  .quad L_.str
  .quad 3
_same_b:
  .quad ___CFConstantStringClassReference@AUTH(da,27361,addr)
  .long 1992
  .space 4
  .quad L_.str
  .quad 3
_other_diversity:
  .quad ___CFConstantStringClassReference@AUTH(da,27362,addr)
  .long 1992
  .space 4
  .quad L_.str
  .quad 3
_unsigned:
  .quad ___CFConstantStringClassReference
  .long 1992
  .space 4
  .quad L_.str
  .quad 3

.section __TEXT,__cstring,cstring_literals
L_.str:
  .asciz "abc"

.data
.globl ___CFConstantStringClassReference
___CFConstantStringClassReference:
  .quad 0

.subsections_via_symbols
