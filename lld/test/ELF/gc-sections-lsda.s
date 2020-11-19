# REQUIRES: x86
# RUN: llvm-mc %s -o %t.o -filetype=obj -triple=x86_64-pc-linux

## Discard an unused .gcc_except_table in a COMDAT group if the associated text
## section is discarded.

# RUN: ld.lld --gc-sections --print-gc-sections -u _Z3foov %t.o -o /dev/null | \
# RUN:   FileCheck %s --implicit-check-not=.gcc_except_table

# CHECK:      removing unused section {{.*}}.o:(.text._Z6comdatv)
# CHECK-NEXT: removing unused section {{.*}}.o:(.gcc_except_table._Z6comdatv)

## An unused non-group .gcc_except_table is not discarded.

# RUN: ld.lld --gc-sections --print-gc-sections -u _Z6comdatv %t.o -o /dev/null | \
# RUN:   FileCheck /dev/null --implicit-check-not=.gcc_except_table

## If the text sections are live, the .gcc_except_table sections are retained as
## well because they are referenced by .eh_frame pieces.

# RUN: ld.lld --gc-sections --print-gc-sections -u _Z3foov -u _Z6comdatv %t.o -o /dev/null | \
# RUN:   FileCheck %s --check-prefix=KEEP

# KEEP-NOT: .gcc_except_table

.section .text._Z3foov,"ax",@progbits
.globl _Z3foov
_Z3foov:
  .cfi_startproc
  ret
  .cfi_lsda 0x1b,.Lexception0
  .cfi_endproc

.section .text._Z6comdatv,"axG",@progbits,_Z6comdatv,comdat
.globl _Z6comdatv
_Z6comdatv:
  .cfi_startproc
  ret
  .cfi_lsda 0x1b,.Lexception1
  .cfi_endproc

.section .gcc_except_table._Z3foov,"a",@progbits
.Lexception0:
  .byte 255

.section .gcc_except_table._Z6comdatv,"aG",@progbits,_Z6comdatv,comdat
.Lexception1:
  .byte 255
