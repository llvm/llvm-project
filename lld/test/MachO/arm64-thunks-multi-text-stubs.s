# REQUIRES: aarch64

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t.o
# RUN: %lld -arch arm64 -U _extern_sym -o %t %t.o
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t | FileCheck %s --implicit-check-not=.thunk.

# CHECK-LABEL: <_foo>:
# CHECK-NEXT:    bl 0x[[#%x,THUNK:]] <_extern_sym.thunk.0>
# CHECK-NEXT:    ret

# CHECK: [[#THUNK]] <_extern_sym.thunk.0>:

# CHECK-LABEL: <_bar>:
# CHECK-NEXT:    bl 0x{{[0-9a-f]+}}
# CHECK-NEXT:    ret

.text

.globl _main
.p2align 2
_main:
  bl _foo
  ret

_spacer0:
.space 0x4000000-8

.globl _foo
.p2align 2
_foo:
  bl _extern_sym
  ret

_spacer1:
.space 0x4000000

.subsections_via_symbols

.section __TEXT,__text_bar,regular,pure_instructions
.globl _bar
.no_dead_strip _bar
.p2align 2
_bar:
  bl _extern_sym
  ret

_spacer2:
.space 0x4000000

.subsections_via_symbols
