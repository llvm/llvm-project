# REQUIRES: ppc
# RUN: echo 'SECTIONS { \
# RUN:       .text_callee 0x10010000 : { *(.text_callee) } \
# RUN:       .text_caller 0x10020000 : { *(.text_caller) } \
# RUN:       }' > %t.script

# RUN: llvm-mc -filetype=obj -triple=powerpc64le %s -o %t.o
# RUN: ld.lld -T %t.script %t.o -o %t
# RUN: llvm-readelf -s %t | FileCheck %s --check-prefix=SYMBOL
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=powerpc64 %s -o %t.o
# RUN: ld.lld -T %t.script %t.o -o %t
# RUN: llvm-readelf -s %t | FileCheck %s --check-prefix=SYMBOL
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=powerpc64 %s -o %t.o
# RUN: ld.lld -T %t.script %t.o -o %t --no-power10-stubs
# RUN: llvm-readelf -s %t | FileCheck %s --check-prefix=SYMBOL
# RUN: llvm-objdump -d --no-show-raw-insn %t \
# RUN: | FileCheck %s

# The point of this test is to make sure that when a function with TOC access
# a local function with st_other=1, a TOC save stub is inserted.

# SYMBOL: Symbol table '.symtab' contains 7 entries
# SYMBOL: 10010000     0 NOTYPE  LOCAL  DEFAULT [<other: 0x20>]   1 callee
# SYMBOL: 10020000     0 NOTYPE  LOCAL  DEFAULT [<other: 0x60>]   2 caller
# SYMBOL: 10020020     0 NOTYPE  LOCAL  DEFAULT [<other: 0x60>]   2 caller_14
# SYMBOL: 10020040     8 FUNC    LOCAL  DEFAULT                   2 __toc_save_callee

# CHECK-LABEL: <callee>:
# CHECK:       blr

# CHECK-LABEL: <caller>:
# CHECK:       bl 0x10020040 <__toc_save_callee>
# CHECK-NEXT:  ld 2, 24(1)
# CHECK-NEXT:  blr

# CHECK-LABEL: <caller_14>:
# CHECK:       bfl 0, 0x10020040 <__toc_save_callee>
# CHECK-NEXT:  ld 2, 24(1)
# CHECK-NEXT:  blr

# CHECK-LABEL: <__toc_save_callee>:
# CHECK-NEXT:  std 2, 24(1)
# CHECK-NEXT:  b 0x10010000 <callee>

.section .text_callee, "ax", %progbits
callee:
  .localentry callee, 1
  blr

.section .text_caller, "ax", %progbits
caller:
.Lfunc_gep1:
  addis 2, 12, .TOC.-.Lfunc_gep1@ha
  addi 2, 2, .TOC.-.Lfunc_gep1@l
.Lfunc_lep1:
  .localentry caller, .Lfunc_lep1-.Lfunc_gep1
  addis 30, 2, global@toc@ha
  lwz 3, global@toc@l(30)
  bl callee
  nop
  blr
global:
  .long	0

caller_14:
.Lfunc_gep2:
  addis 2, 12, .TOC.-.Lfunc_gep1@ha
  addi 2, 2, .TOC.-.Lfunc_gep1@l
.Lfunc_lep2:
  .localentry caller_14, .Lfunc_lep2-.Lfunc_gep2
  addis 30, 2, global@toc@ha
  lwz 3, global@toc@l(30)
  bcl 4, 0, callee
  nop
  blr
