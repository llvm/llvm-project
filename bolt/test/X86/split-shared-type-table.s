## Check that a function split into several fragments emits a single copy of
## the LSDA type table. @TType base is an unsigned forward offset from each
## LSDA header, so one table placed after the last LSDA serves every fragment.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-linux %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: ld.lld --no-pie %t.o -o %t.exe -q -e _start
# RUN: llvm-bolt %t.exe -o %t.bolt --data %t.fdata --split-functions --split-eh \
# RUN:   --split-all-cold

## _start keeps a call site in either fragment, so both get an LSDA.
# RUN: llvm-bolt %t.exe -o %t.null --data %t.fdata --split-functions --split-eh \
# RUN:   --split-all-cold --print-after-lowering --print-only=_start 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-SPLIT
# CHECK-SPLIT: callq foo # handler: [[LP0:[^;]+]]; action: 1
# CHECK-SPLIT: HOT-COLD SPLIT POINT
# CHECK-SPLIT: callq foo # handler: [[LP1:[^;]+]]; action: 3

## Yet the type table is emitted once: each entry appears exactly once in
## .gcc_except_table. TI1 is at 0x200120 and TI2 at 0x200128.
# RUN: llvm-objdump -s -j .gcc_except_table %t.bolt \
# RUN:   | FileCheck %s --check-prefix=CHECK-TT \
# RUN:     --implicit-check-not=20012000 --implicit-check-not=28012000
# CHECK-TT: 28012000 20012000

  .text
  .globl foo
  .type foo, %function
foo:
  .cfi_startproc
  ret
  .cfi_endproc
  .size foo, .-foo

  .globl _start
  .type _start, %function
_start:
# FDATA: 0 [unknown] 0 1 _start 0 1 100
.Lfunc_begin0:
  .cfi_startproc
  .cfi_lsda 27, .Lexception0
## Hot call site, in the entry block, which is never outlined.
Lhot:
  call foo
.Ltmp0:
  cmpl $0, %edi
  jne Lcold
  ret
## Cold call site, with a different catch type. Padded so that splitting the
## function is profitable and does not get undone.
Lcold:
  call foo
.Ltmp1:
  .rept 64
  addl $1, %eax
  .endr
  ret
.LLP0:
  ret
.LLP1:
  ret
  .cfi_endproc
.Lfunc_end0:
  .size _start, .-_start

## Two fake typeinfo objects.
  .section .rodata,"a",@progbits
  .p2align 3
  .globl TI1
TI1:
  .quad 0x1111111111111111
  .globl TI2
TI2:
  .quad 0x2222222222222222

## EH table.
  .section .gcc_except_table,"a",@progbits
  .p2align 2
GCC_except_table0:
.Lexception0:
  .byte 255                             # @LPStart Encoding = omit
  .byte 3                               # @TType Encoding = udata4
  .uleb128 .Lttbase0-.Lttbaseref0
.Lttbaseref0:
  .byte 1                               # Call site Encoding = uleb128
  .uleb128 .Lcst_end0-.Lcst_begin0
.Lcst_begin0:
  .uleb128 Lhot-.Lfunc_begin0           # >> Call Site 1 <<
  .uleb128 .Ltmp0-Lhot
  .uleb128 .LLP0-.Lfunc_begin0
  .byte 1                               #   On action: catch TI1
  .uleb128 Lcold-.Lfunc_begin0          # >> Call Site 2 <<
  .uleb128 .Ltmp1-Lcold
  .uleb128 .LLP1-.Lfunc_begin0
  .byte 3                               #   On action: catch TI2
.Lcst_end0:
  .byte 1, 0                            # >> Action Record 1 << catch TypeInfo 1
  .byte 2, 0                            # >> Action Record 2 << catch TypeInfo 2
  .p2align 2
  .long TI2                             # TypeInfo 2
  .long TI1                             # TypeInfo 1
.Lttbase0:
  .p2align 2
