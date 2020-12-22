# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
## Verify that we preserve alignment when merging sections.
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/bar.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/baz.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/qux.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/main.o
# RUN: %lld -o %t/output %t/foo.o %t/bar.o %t/baz.o %t/qux.o %t/main.o

# RUN: llvm-objdump --syms --section=__data --full-contents %t/output | FileCheck %s
# CHECK:     SYMBOL TABLE:
# CHECK-DAG: [[#%x, ADDR:]]      g     O __DATA,__data _foo
# CHECK-DAG: {{0*}}[[#ADDR+0x4]] g     O __DATA,__data _bar
# CHECK-DAG: {{0*}}[[#ADDR+0x8]] g     O __DATA,__data _baz
# CHECK-DAG: {{0*}}[[#ADDR+0x9]] g     O __DATA,__data _qux

# CHECK:      Contents of section __DATA,__data:
# CHECK-NEXT: {{0*}}[[#ADDR]] ca000000 fe000000 baefbead de000000

#--- foo.s
.globl _foo
.data
.p2align 0
_foo:
  .byte 0xca

#--- bar.s
.globl _bar
.data
.p2align 2
_bar:
  .byte 0xfe

#--- baz.s
.globl _baz
.data
.p2align 3
_baz:
  .byte 0xba

#--- qux.s
.globl _qux
.data
.p2align 0
_qux:
  .quad 0xdeadbeef

#--- main.s
.section __TEXT,__text
.globl _main

_main:
  mov $0, %rax
  ret
