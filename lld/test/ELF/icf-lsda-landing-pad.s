# REQUIRES: x86
## LSDA sections that relocate to their function's landing pad are compared
## with relocations taken into account: two .gcc_except_table sections with the
## same bytes are equivalent only if their relocation targets are equivalent.
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld --icf=all --print-icf-sections %t.o -o /dev/null | \
# RUN:   FileCheck %s --implicit-check-not=Z1cv --implicit-check-not=Z1dv

## _Z1av and _Z1bv have identical code and their LSDA sections point to
## equivalent landing pads: fold.
# CHECK-DAG: selected section {{.*}}.o:(.text.Z1av)
# CHECK-DAG: removing identical section {{.*}}.o:(.text.Z1bv)

.globl _Z1av, _Z1bv, _Z1cv, _Z1dv
.section .text.Z1av,"ax",@progbits
_Z1av:
  .cfi_startproc
  .cfi_lsda 27, .Llsda_a
  ret
  .cfi_endproc
.section .text.Z1bv,"ax",@progbits
_Z1bv:
  .cfi_startproc
  .cfi_lsda 27, .Llsda_b
  ret
  .cfi_endproc

## _Z1cv and _Z1dv have identical code, but their LSDA sections point to
## different landing pads: do not fold.
.section .text.Z1cv,"ax",@progbits
_Z1cv:
  .cfi_startproc
  .cfi_lsda 27, .Llsda_c
  ret
  .cfi_endproc
.section .text.Z1dv,"ax",@progbits
_Z1dv:
  .cfi_startproc
  .cfi_lsda 27, .Llsda_d
  ret
  .cfi_endproc

.section .text.lpad.a,"ax",@progbits
.Lpad_a:
  ret
.section .text.lpad.b,"ax",@progbits
.Lpad_b:
  ret
.section .text.lpad.c,"ax",@progbits
.Lpad_c:
  nop
  ret
.section .text.lpad.d,"ax",@progbits
.Lpad_d:
  nop
  nop
  ret

.section .gcc_except_table.a,"a",@progbits
.Llsda_a:
  .long .Lpad_a - .
.section .gcc_except_table.b,"a",@progbits
.Llsda_b:
  .long .Lpad_b - .
.section .gcc_except_table.c,"a",@progbits
.Llsda_c:
  .long .Lpad_c - .
.section .gcc_except_table.d,"a",@progbits
.Llsda_d:
  .long .Lpad_d - .
