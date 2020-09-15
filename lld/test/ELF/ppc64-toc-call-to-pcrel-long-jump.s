# REQUIRES: ppc
# RUN: echo 'SECTIONS { \
# RUN:       .text_callee 0x10010000 : { *(.text_callee) } \
# RUN:       .text_caller 0x20020000 : { *(.text_caller) } \
# RUN:       }' > %t.script

# RUN: llvm-mc -filetype=obj -triple=powerpc64le %s -o %t.o
# RUN: not ld.lld -T %t.script %t.o -o /dev/null 2>&1 | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=powerpc64 %s -o %t.o
# RUN: not ld.lld -T %t.script %t.o -o /dev/null 2>&1 | FileCheck %s

# CHECK:      error: R2 save stub offset is out of range: -268501028 is not in [-33554432, 33554431]; references callee
# CHECK-NEXT: >>> defined in {{.*}}.o

# RUN: ld.lld -T %t.script %t.o -o /dev/null --noinhibit-exec

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
