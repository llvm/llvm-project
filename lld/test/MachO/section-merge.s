# REQUIRES: x86
# RUN: mkdir -p %t
## Verify that we preserve alignment when merging sections.
# RUN: echo ".globl _foo; .data; .p2align 0; _foo: .byte 0xca" | llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/foo.o
# RUN: echo ".globl _bar; .data; .p2align 2; _bar: .byte 0xfe" | llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/bar.o
# RUN: echo ".globl _baz; .data; .p2align 3; _baz: .byte 0xba" | llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/baz.o
# RUN: echo ".globl _qux; .data; .p2align 0; _qux: .quad 0xdeadbeef" | llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/qux.o
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

.section __TEXT,__text
.global _main

_main:
  mov $0, %rax
  ret
