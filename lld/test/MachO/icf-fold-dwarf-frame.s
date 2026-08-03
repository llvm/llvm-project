# REQUIRES: aarch64

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos11 %s -o %t.o
# RUN: %lld -arch arm64 %t.o -o %t.dylib -undefined dynamic_lookup --icf=all
# RUN: llvm-objdump --macho --syms --dwarf=frames --unwind-info %t.dylib | FileCheck %s

# CHECK-LABEL: SYMBOL TABLE:
# CHECK-DAG:  [[#%.16x,FOLD0:]] {{.*}} __TEXT,__text _foldA
# CHECK-DAG:  [[#FOLD0]]        {{.*}} __TEXT,__text _foldB
# CHECK-DAG:  [[#FOLD0]]        {{.*}} __TEXT,__text _foldC
# CHECK-NOT:  [[#FOLD0]]        {{.*}} __TEXT,__text _diffLSDA
# CHECK-NOT:  [[#FOLD0]]        {{.*}} __TEXT,__text _diffPersonality
# CHECK-NOT:  [[#FOLD0]]        {{.*}} __TEXT,__text _diffCFI
# CHECK-DAG:  [[#%.16x,FOLD1:]] {{.*}} __TEXT,__gcc_except_tab GCC_except_table0
# CHECK-DAG:  [[#FOLD1]]        {{.*}} __TEXT,__gcc_except_tab GCC_except_table1
# CHECK-NOT:  [[#FOLD1]]        {{.*}} __TEXT,__gcc_except_tab GCC_except_table2

# CHECK-LABEL: Contents of __unwind_info section:

# CHECK-LABEL: .eh_frame contents:
# CHECK: {{^}}[[#%.8x,CIE0:]] {{.*}} CIE
# CHECK: FDE cie=[[#%.8x,CIE0]]
# CHECK: FDE cie=[[#%.8x,CIE0]]
# CHECK: FDE cie=[[#%.8x,CIE0]]
# CHECK: FDE cie=[[#%.8x,CIE0]]
# CHECK: {{^}}[[#%.8x,CIE1:]] {{.*}} CIE
# CHECK: FDE cie=[[#%.8x,CIE1]]
# CHECK: FDE cie=[[#%.8x,CIE1]]

# Due to padding, we need to emit a throwaway FDE for each personality, so that
# subsequent FDEs are the same size and can be folded
# TODO: Could we detect padding differences and fold anyway?
_padA:
  .cfi_startproc
  .cfi_personality 155, _p0
  .cfi_lsda 16, Lexception0
  str x30, [sp, #-16]!
  .cfi_def_cfa_offset 16
  .cfi_offset w30, -16
  bl _may_throw
  ldr x30, [sp], #16
  ret
  .cfi_endproc

_padB:
  .cfi_startproc
  .cfi_personality 155, _p1
  .cfi_lsda 16, Lexception0
  str x30, [sp, #-16]!
  .cfi_def_cfa_offset 16
  .cfi_offset w30, -16
  bl _may_throw
  ldr x30, [sp], #16
  ret
  .cfi_endproc

_foldA:
  .cfi_startproc
  .cfi_personality 155, _p0
  .cfi_lsda 16, Lexception0
  str x30, [sp, #-16]!
  .cfi_def_cfa_offset 16
  .cfi_offset w30, -16
  bl _may_throw
  ldr x30, [sp], #16
  ret
  .cfi_endproc

_foldB:
  .cfi_startproc
  .cfi_personality 155, _p0
  .cfi_lsda 16, Lexception0
  str x30, [sp, #-16]!
  .cfi_def_cfa_offset 16
  .cfi_offset w30, -16
  bl _may_throw
  ldr x30, [sp], #16
  ret
  .cfi_endproc

# LSDA points to a different symbol, but the contents are the same as Lexception1 so it can be folded
_foldC:
  .cfi_startproc
  .cfi_personality 155, _p0
  .cfi_lsda 16, Lexception1
  str x30, [sp, #-16]!
  .cfi_def_cfa_offset 16
  .cfi_offset w30, -16
  bl _may_throw
  ldr x30, [sp], #16
  ret
  .cfi_endproc

# Different LSDAs cannot be folded
_diffLSDA:
  .cfi_startproc
  .cfi_personality 155, _p0
  .cfi_lsda 16, Lexception2
  str x30, [sp, #-16]!
  .cfi_def_cfa_offset 16
  .cfi_offset w30, -16
  bl _may_throw
  ldr x30, [sp], #16
  ret
  .cfi_endproc

_diffPersonality:
  .cfi_startproc
  .cfi_personality 155, _p1
  .cfi_lsda 16, Lexception0
  str x30, [sp, #-16]!
  .cfi_def_cfa_offset 16
  .cfi_offset w30, -16
  bl _may_throw
  ldr x30, [sp], #16
  ret
  .cfi_endproc

_diffCFI:
  .cfi_startproc
  .cfi_personality 155, _p0
  .cfi_lsda 16, Lexception0
  str x30, [sp, #-16]!
  .cfi_def_cfa_offset 160
  .cfi_offset w30, -160
  bl _may_throw
  ldr x30, [sp], #16
  ret
  .cfi_endproc

.section __TEXT,__gcc_except_tab
.p2align 2
GCC_except_table0:
Lexception0:
  .byte 0xFF
  .byte 0x9B
  .uleb128 Lttbase0-Lttbaseref0
Lttbaseref0:
  .byte 1
  .uleb128 0
  .p2align 2
  .long 0
Lttbase0:

# Contents are identical to above. It should be folded
.p2align 2
GCC_except_table1:
Lexception1:
  .byte 0xFF
  .byte 0x9B
  .uleb128 Lttbase1-Lttbaseref1
Lttbaseref1:
  .byte 1
  .uleb128 0
  .p2align 2
  .long 0
Lttbase1:

.p2align 2
GCC_except_table2:
Lexception2:
  .byte 0xFF
  .byte 0x9B
  .uleb128 Lttbase2-Lttbaseref2
Lttbaseref2:
  .byte 1
  .uleb128 1
  .byte 0xAA
  .p2align 2
  .long 0
Lttbase2:

.subsections_via_symbols
