# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t

## This tests that if two input files define the same weak symbol, we only
## write it to the output once (...assuming both input files use
## .subsections_via_symbols).

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/weak-sub.s -o %t/weak-sub.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/weak-nosub.s -o %t/weak-nosub.o

## Test that weak symbols are emitted just once with .subsections_via_symbols
# RUN: %lld -dylib -o %t/out.dylib %t/weak-sub.o %t/weak-sub.o
# RUN: llvm-otool -jtV %t/out.dylib | FileCheck --check-prefix=SUB %s
# RUN: %lld -dylib -o %t/out.dylib %t/weak-nosub.o %t/weak-sub.o
# RUN: llvm-otool -jtV %t/out.dylib | FileCheck --check-prefix=SUB %s
# RUN: %lld -dylib -o %t/out.dylib %t/weak-sub.o %t/weak-nosub.o
# RUN: llvm-otool -jtV %t/out.dylib | FileCheck --check-prefix=SUB %s
# SUB:      _foo
# SUB-NEXT: retq
# SUB-NOT:  retq
# SUB:      _bar
# SUB-NEXT: retq
# SUB-NOT:  retq

## We can even strip weak symbols without subsections_via_symbols as long
## as none of the weak symbols in a section are needed.
# RUN: %lld -dylib -o %t/out.dylib %t/weak-nosub.o %t/weak-nosub.o
# RUN: llvm-otool -jtV %t/out.dylib | FileCheck --check-prefix=SUB %s

## Test that omitted weak symbols don't add entries to the compact unwind table.
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 %t/weak-sub-lsda.s -o %t/weak-sub-lsda.o
# RUN: %lld -dylib -lc++ -o %t/out.dylib %t/weak-sub-lsda.o %t/weak-sub-lsda.o
# RUN: llvm-objdump --macho --unwind-info --syms %t/out.dylib | FileCheck %s --check-prefix=UNWIND -D#%x,BASE=0

# UNWIND:      SYMBOL TABLE:
# UNWIND-DAG:  [[#%x,FOO:]]       w  F __TEXT,__text _foo
# UNWIND-NOT:                          __TEXT,__text _foo

# UNWIND:      Contents of __unwind_info section:
# UNWIND:        LSDA descriptors:
# UNWIND:           [0]: function offset=0x[[#%.8x,FOO-BASE]]
# UNWIND-NOT:       [1]:
# UNWIND:        Second level indices:
# UNWIND-DAG:      [0]: function offset=0x[[#%.8x,FOO-BASE]]
# UNWIND-NOT:      [1]:

## Test interaction with .alt_entry
## FIXME: ld64 manages to strip both one copy of _foo and _bar each.
##        We only manage this if we're lucky and the object files are in
##        the right order. We're happy to not crash at link time for now.
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 %t/weak-sub-alt.s -o %t/weak-sub-alt.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 %t/weak-sub-alt2.s -o %t/weak-sub-alt2.o
# RUN: %lld -dylib -o %t/out.dylib %t/weak-sub-alt.o %t/weak-sub-alt2.o
# RUN: %lld -dylib -o %t/out.dylib %t/weak-sub-alt2.o %t/weak-sub-alt.o

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 %t/weak-aligned-1.s -o %t/weak-aligned-1.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 %t/weak-aligned-2.s -o %t/weak-aligned-2.o
# RUN: %lld -o %t/out -lSystem %t/weak-aligned-1.o %t/weak-aligned-2.o
# RUN: llvm-objdump --syms --section=__const --full-contents %t/out | FileCheck --check-prefix=ALIGN %s
# ALIGN:      SYMBOL TABLE:
# ALIGN-DAG:  [[#%x, ADDR:]]       l     O __DATA_CONST,__const _weak1
# ALIGN-DAG:  {{0*}}[[#ADDR+ 0x4]] l     O __DATA_CONST,__const _weak3
# ALIGN-DAG:  {{0*}}[[#ADDR+ 0x8]] l     O __DATA_CONST,__const _weak2
# ALIGN-DAG:  {{0*}}[[#ADDR+0x10]] g     O __DATA_CONST,__const _aligned
# ALIGN:      Contents of section __DATA_CONST,__const:
# ALIGN-NEXT: {{0*}}[[#ADDR]]      11111111 33333333 22222222 00000000
# ALIGN-NEXT: {{0*}}[[#ADDR+0x10]] 81818181 81818181 82828282 82828282

#--- weak-sub.s
.globl _foo, _bar
.weak_definition _foo, _bar
_foo:
  retq
_bar:
  retq
.subsections_via_symbols

#--- weak-nosub.s
.globl _foo, _bar
.weak_definition _foo, _bar
_foo:
  retq
_bar:
  retq

#--- weak-sub-lsda.s
.section __TEXT,__text,regular,pure_instructions

.globl _foo
.weak_definition _foo
_foo:
  .cfi_startproc
  .cfi_personality 155, ___gxx_personality_v0
  .cfi_lsda 16, Lexception
  pushq %rbp
  .cfi_def_cfa_offset 128
  .cfi_offset %rbp, 48
  movq %rsp, %rbp
  .cfi_def_cfa_register %rbp
  popq %rbp
  retq
  .cfi_endproc

.section __TEXT,__gcc_except_tab
Lexception:
    .space 0x10

.subsections_via_symbols

#--- weak-sub-alt.s
.globl _foo, _bar
.weak_definition _foo
_foo:
  retq

# Alternative entry point to _foo (strong)
.alt_entry _bar
_bar:
  retq

.globl _main, _ref
_main:
  callq _ref
  callq _bar

.subsections_via_symbols

#--- weak-sub-alt2.s
.globl _foo, _bar
.weak_definition _foo
_foo:
  retq

# Alternative entry point to _foo (weak)
.weak_definition _bar
.alt_entry _bar
_bar:
  retq

.globl _ref
_ref:
  callq _bar

.subsections_via_symbols

#--- weak-aligned-1.s
.section __DATA,__const
.p2align 3
.globl _weak1
.weak_def_can_be_hidden _weak1
_weak1:
  .4byte 0x11111111

.globl _weak3
.weak_def_can_be_hidden _weak3
_weak3:
  .4byte 0x33333333

.subsections_via_symbols

#--- weak-aligned-2.s
# _weak1 and _weak3 are already in weak-aligned-1,
# so from _weak1-3 in this file only _weak2 is used.
# However, _aligned still has to stay aligned to a 16-byte boundary.
.section __DATA,__const
.p2align 3
.globl _weak1
.weak_def_can_be_hidden _weak1
_weak1:
  .4byte 0x11111111

.globl _weak2
.weak_def_can_be_hidden _weak2
_weak2:
  .4byte 0x22222222

.globl _weak3
.weak_def_can_be_hidden _weak3
_weak3:
  .4byte 0x33333333

.section __DATA,__const
.p2align 4
.globl _aligned
_aligned:
  .8byte 0x8181818181818181
  .8byte 0x8282828282828282

.section __TEXT,__text
.globl _main
_main:
  movl _weak1(%rip), %eax
  movl _weak2(%rip), %ebx
  movaps _aligned(%rip), %xmm0
  retq

.subsections_via_symbols
