# REQUIRES: x86
## An FDE with more than one relocation after the PC-begin relocation cannot be
## analyzed. LSDA-aware ICF must conservatively keep the function unique rather
## than treating it as a function without an LSDA. RISC-V linker relaxation
## (R_RISCV_ADD32/R_RISCV_SUB32 on the PC range) is a realistic producer of
## such FDEs.
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld --icf=all --print-icf-sections %t.o -o /dev/null | \
# RUN:   FileCheck %s --implicit-check-not=Z1av --implicit-check-not=Z1bv

## _Z1av and _Z1bv have identical code and different LSDA, but their FDEs carry
## an extra relocation, so they must not fold. _Z1cv and _Z1dv exercise the same
## hand-written .eh_frame shape without the extra relocation: their LSDA is
## identical and they fold.
# CHECK-DAG: selected section {{.*}}.o:(.text.Z1cv)
# CHECK-DAG: removing identical section {{.*}}.o:(.text.Z1dv)

.globl _Z1av, _Z1bv, _Z1cv, _Z1dv
.section .text.Z1av,"ax",@progbits
_Z1av:
  ret
.section .text.Z1bv,"ax",@progbits
_Z1bv:
  ret
.section .text.Z1cv,"ax",@progbits
_Z1cv:
  ret
.section .text.Z1dv,"ax",@progbits
_Z1dv:
  ret

.section .gcc_except_table.a,"a",@progbits
.Llsda_a:
  .long 0x11111111
.section .gcc_except_table.b,"a",@progbits
.Llsda_b:
  .long 0x22222222
.section .gcc_except_table.c,"a",@progbits
.Llsda_c:
  .long 0x33333333
.section .gcc_except_table.d,"a",@progbits
.Llsda_d:
  .long 0x33333333

## The extra relocation target.
.section .rodata.extra,"a",@progbits
extra:
  .byte 0

.section .text.per,"ax",@progbits
per:
  ret

## Hand-written .eh_frame: an FDE with an extra relocation cannot be produced
## with .cfi_lsda alone.
.section .eh_frame,"a",@progbits
.Lcie:
  .long .Lcie_end - .Lcie_begin
.Lcie_begin:
  .long 0                   # CIE id
  .byte 1                   # version
  .asciz "zPLR"
  .uleb128 1                # code alignment
  .sleb128 -8               # data alignment
  .byte 16                  # return address register
  .uleb128 7                # augmentation length
  .byte 0x1b                # personality encoding
  .long per - .
  .byte 0x1b                # LSDA encoding
  .byte 0x1b                # FDE encoding
.Lcie_end:

## _Z1av: PC-begin, LSDA and an extra relocation.
.Lfde_a:
  .long .Lfde_a_end - .Lfde_a_begin
.Lfde_a_begin:
  .long .Lfde_a_begin - .Lcie
  .long _Z1av - .           # PC begin
  .long 1                   # range
  .uleb128 4                # augmentation length
  .long .Llsda_a - .        # LSDA
  .long extra - .           # extra relocation
.Lfde_a_end:

## _Z1bv: PC-begin, LSDA and an extra relocation.
.Lfde_b:
  .long .Lfde_b_end - .Lfde_b_begin
.Lfde_b_begin:
  .long .Lfde_b_begin - .Lcie
  .long _Z1bv - .
  .long 1
  .uleb128 4
  .long .Llsda_b - .
  .long extra - .
.Lfde_b_end:

## _Z1cv and _Z1dv: PC-begin and LSDA only.
.Lfde_c:
  .long .Lfde_c_end - .Lfde_c_begin
.Lfde_c_begin:
  .long .Lfde_c_begin - .Lcie
  .long _Z1cv - .
  .long 1
  .uleb128 4
  .long .Llsda_c - .
.Lfde_c_end:

.Lfde_d:
  .long .Lfde_d_end - .Lfde_d_begin
.Lfde_d_begin:
  .long .Lfde_d_begin - .Lcie
  .long _Z1dv - .
  .long 1
  .uleb128 4
  .long .Llsda_d - .
.Lfde_d_end:
