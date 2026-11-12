# REQUIRES: x86
## Test that text sections with LSDA are not folded.

## Test REL.
# RUN: llvm-mc -filetype=obj -triple=i386 %s -o %t1.o
# RUN: ld.lld --icf=all %t1.o -o /dev/null --print-icf-sections | FileCheck %s --implicit-check-not=removing
## Test RELA.
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t2.o
# RUN: ld.lld --icf=all %t2.o -o /dev/null --print-icf-sections | FileCheck %s --implicit-check-not=removing

# CHECK:      selected section {{.*}}.o:(.text.Z1cv)
# CHECK-NEXT:   removing identical section {{.*}}.o:(.text.Z1dv)

.globl _Z1av, _Z1bv, _Z1cv, _Z1dv
.section .text.Z1av,"ax",@progbits
_Z1av:
  .cfi_startproc
  .cfi_lsda 27, .Lexception0
  ret
  .cfi_endproc

.section .text.Z1bv,"ax",@progbits
_Z1bv:
  .cfi_startproc
  .cfi_lsda 27, .Lexception0
  ret
  .cfi_endproc

.section .text.Z1cv,"ax",@progbits
_Z1cv:
  .cfi_startproc
  .cfi_signal_frame
  ret
  .cfi_endproc

.section .text.Z1dv,"ax",@progbits
_Z1dv:
  .cfi_startproc
  ret
  .cfi_endproc

.section .gcc_except_table,"a",@progbits
## The actual content does not matter.
.Lexception0:

## .rodata.Z1[ab]v reference .text.Z1[ab]v. Dont fold them.
.section .rodata.Z1av,"a",@progbits
  .long .text.Z1av - .

.section .rodata.Z1bv,"a",@progbits
  .long .text.Z1bv - .
