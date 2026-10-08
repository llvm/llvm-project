# REQUIRES: aarch64

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t.o
# RUN: %lld -arch arm64 -U _extern_sym -o %t %t.o
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t | FileCheck %s --implicit-check-not=.thunk.

# CHECK-LABEL: Disassembly of section __TEXT,__text:

# CHECK-LABEL: <_main>:
# CHECK-NEXT:    bl
# CHECK-NEXT:    bl 0x[[#%x,THUNK:]] <_extern_sym.thunk.0>
# CHECK-NEXT:    ret

# CHECK-LABEL: <_foo>:
# CHECK-NEXT:    bl 0x[[#%x,THUNK:]] <_extern_sym.thunk.0>
# CHECK-NEXT:    ret

# CHECK: [[#THUNK]] <_extern_sym.thunk.0>:

# CHECK-LABEL: Disassembly of section __TEXT,__lcxx_override:
# CHECK-LABEL: Disassembly of section __TEXT,__stubs:

.text

.globl _main
_main:
  bl _foo
  bl _extern_sym
  ret

_spacer0:
.space 0x4000000-8

.globl _foo
_foo:
  bl _extern_sym
  ret

_spacer1:
.space 0x4000000

.section __TEXT,__lcxx_override,regular,pure_instructions
_bar:
  bl _extern_sym
  ret

_spacer2:
.space 0x4000000

.subsections_via_symbols
