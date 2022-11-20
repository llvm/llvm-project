# REQUIRES: aarch64

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64_32-apple-darwin %t/main.s -o %t/main.o
# RUN: llvm-mc -filetype=obj -triple=arm64_32-apple-darwin %t/foobar.s -o %t/foobar.o

# RUN: %lld-watchos -lSystem -arch arm64_32 -o %t/static %t/main.o %t/foobar.o
# RUN: llvm-objdump --no-print-imm-hex --macho -d --no-show-raw-insn --syms %t/static | FileCheck %s --check-prefix=STATIC

# RUN: %lld-watchos -lSystem -arch arm64_32 -dylib -o %t/libfoo.dylib %t/foobar.o
# RUN: %lld-watchos -lSystem -arch arm64_32 -o %t/main %t/main.o %t/libfoo.dylib
# RUN: llvm-objdump --no-print-imm-hex --macho -d --no-show-raw-insn --section-headers %t/main | FileCheck %s --check-prefix=DYLIB

# STATIC-LABEL: _main:
# STATIC-NEXT:  adrp x8, [[#]] ; 0x[[#%x,PAGE:]]
# STATIC-NEXT:  add  x8, x8, #[[#%u,FOO_OFF:]]
# STATIC-NEXT:  adrp x8, [[#]] ; 0x[[#PAGE]]
# STATIC-NEXT:  add  x8, x8, #[[#%u,BAR_OFF:]]
# STATIC-NEXT:  ret

# STATIC-LABEL: SYMBOL TABLE:
# STATIC-DAG:   {{0*}}[[#%x,PAGE+FOO_OFF]] g     F __TEXT,__text _foo
# STATIC-DAG:   {{0*}}[[#%x,PAGE+BAR_OFF]] g     F __TEXT,__text _bar

# DYLIB-LABEL: _main:
# DYLIB-NEXT:  adrp x8, [[#]] ; 0x[[#%x,GOT:]]
# DYLIB-NEXT:  ldr  w8, [x8, #4]
# DYLIB-NEXT:  adrp x8, [[#]] ; 0x[[#GOT]]
# DYLIB-NEXT:  ldr  w8, [x8]
# DYLIB-NEXT:  ret
# DYLIB-EMPTY:
# DYLIB-NEXT:  Sections:
# DYLIB-NEXT:  Idx   Name   Size      VMA            Type
# DYLIB:       [[#]] __got  00000008  [[#%.8x,GOT]]  DATA

#--- main.s
.globl _main, _foo, _bar
.p2align 2
_main:
  adrp x8, _foo@GOTPAGE
  ldr  w8, [x8, _foo@GOTPAGEOFF]
  adrp x8, _bar@GOTPAGE
  ldr  w8, [x8, _bar@GOTPAGEOFF]
  ret

#--- foobar.s
.globl _foo, _bar
_foo:
_bar:
