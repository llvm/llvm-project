# REQUIRES: x86
## Test LSDA-aware ICF:
## * text sections with the same LSDA are folded;
## * text sections with different LSDA are not folded;
## * text sections whose LSDA is at a nonzero offset in a shared
##   .gcc_except_table section are not folded.

## Test REL.
# RUN: llvm-mc -filetype=obj -triple=i386 %s -o %t1.o
# RUN: ld.lld --icf=all %t1.o -o /dev/null --print-icf-sections | FileCheck %s --implicit-check-not=Z1[abgh]v
## Test RELA.
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t2.o
# RUN: ld.lld --icf=all %t2.o -o /dev/null --print-icf-sections | FileCheck %s --implicit-check-not=Z1[abgh]v

# CHECK-DAG: selected section {{.*}}.o:(.text.Z1cv)
# CHECK-DAG: removing identical section {{.*}}.o:(.text.Z1dv)
# CHECK-DAG: selected section {{.*}}.o:(.text.Z1ev)
# CHECK-DAG: removing identical section {{.*}}.o:(.text.Z1fv)

## Z1av and Z1bv have identical code but different LSDA sections.
.globl _Z1av, _Z1bv
.section .text.Z1av,"ax",@progbits
_Z1av:
  .cfi_startproc
  .cfi_lsda 27, .Lexception0
  ret
  .cfi_endproc

.section .text.Z1bv,"ax",@progbits
_Z1bv:
  .cfi_startproc
  .cfi_lsda 27, .Lexception1
  ret
  .cfi_endproc

## Z1cv and Z1dv share the same LSDA and are folded.
.globl _Z1cv, _Z1dv
.section .text.Z1cv,"ax",@progbits
_Z1cv:
  .cfi_startproc
  .cfi_lsda 27, .Lexception2
  ret
  .cfi_endproc

.section .text.Z1dv,"ax",@progbits
_Z1dv:
  .cfi_startproc
  .cfi_lsda 27, .Lexception2
  ret
  .cfi_endproc

## Z1ev and Z1fv have no LSDA and are folded.
.globl _Z1ev, _Z1fv
.section .text.Z1ev,"ax",@progbits
_Z1ev:
  .cfi_startproc
  ret
  .cfi_endproc

.section .text.Z1fv,"ax",@progbits
_Z1fv:
  .cfi_startproc
  ret
  .cfi_endproc

## Z1gv and Z1hv have identical code but their LSDAs are at different
## offsets of the same .gcc_except_table section, so they are not eligible.
.globl _Z1gv, _Z1hv
.section .text.Z1gv,"ax",@progbits
_Z1gv:
  .cfi_startproc
  .cfi_lsda 27, .Lexception3
  ret
  .cfi_endproc

.section .text.Z1hv,"ax",@progbits
_Z1hv:
  .cfi_startproc
  .cfi_lsda 27, .Lexception3b
  ret
  .cfi_endproc

.section .gcc_except_table.0,"a",@progbits
.Lexception0:
  .long 0

.section .gcc_except_table.1,"a",@progbits
.Lexception1:
  .long 1

.section .gcc_except_table.2,"a",@progbits
.Lexception2:
  .long 2

.section .gcc_except_table.3,"a",@progbits
.Lexception3:
  .long 3
.Lexception3b:
  .long 4
