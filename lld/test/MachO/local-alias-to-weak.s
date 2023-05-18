# REQUIRES: x86
## This test checks that when we coalesce weak definitions, their local symbol
## aliases defs don't cause the coalesced data to be retained. This was
## motivated by MC's aarch64 backend which automatically creates `ltmp<N>`
## symbols at the start of each .text section. These symbols are frequently
## aliases of other symbols created by clang or other inputs to MC. I've chosen
## to explicitly create them here since we can then reference those symbols for
## a more complete test.
##
## Not retaining the data matters for more than just size -- we have a use case
## that depends on proper data coalescing to emit a valid file format. We also
## need this behavior to properly deduplicate the __objc_protolist section;
## failure to do this can result in dyld crashing on iOS 13.
##
## Finally, ld64 does all this regardless of whether .subsections_via_symbols is
## specified. We don't. But again, given how rare the lack of that directive is
## (I've only seen it from hand-written assembly inputs), I don't think we need
## to worry about it.

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/weak-then-local.s -o %t/weak-then-local.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/local-then-weak.s -o %t/local-then-weak.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/no-subsections.s -o %t/no-subsections.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/no-dead-strip.s -o %t/no-dead-strip.o

# RUN: %lld -lSystem -dylib %t/weak-then-local.o %t/local-then-weak.o -o %t/test1
# RUN: llvm-objdump --macho --syms --section="__DATA,__data" --weak-bind %t/test1 | FileCheck %s
# RUN: %lld -lSystem -dylib %t/local-then-weak.o %t/weak-then-local.o -o %t/test2
# RUN: llvm-objdump --macho --syms --section="__DATA,__data" --weak-bind %t/test2 | FileCheck %s

## Check that we only have one copy of 0x123 in the data, not two.
# CHECK:       Contents of (__DATA,__data) section
# CHECK-NEXT:  0000000000001000  23 01 00 00 00 00 00 00 00 10 00 00 00 00 00 00 {{$}}
# CHECK-NEXT:  0000000000001010  00 10 00 00 00 00 00 00 {{$}}
# CHECK-EMPTY:
# CHECK-NEXT:  SYMBOL TABLE:
# CHECK-NEXT:  0000000000001000 l     O __DATA,__data _alias
# CHECK-NEXT:  0000000000001008 l     O __DATA,__data _ref
# CHECK-NEXT:  0000000000001000 l     O __DATA,__data _alias
# CHECK-NEXT:  0000000000001010 l     O __DATA,__data _ref
# CHECK-NEXT:  0000000000001000  w    O __DATA,__data _weak
# CHECK-NEXT:  0000000000000000         *UND* dyld_stub_binder
# CHECK-EMPTY:
## Even though the references were to the non-weak `_alias` symbols, ld64 still
## emits weak binds as if they were the `_weak` symbol itself. We do not. I
## don't know of any programs that rely on this behavior, so I'm just
## documenting it here.
# CHECK-NEXT:  Weak bind table:
# CHECK-NEXT:  segment  section            address     type       addend   symbol
# CHECK-EMPTY:

# RUN: %lld -lSystem -dylib %t/local-then-weak.o %t/no-subsections.o -o %t/sub-nosub
# RUN: llvm-objdump --macho --syms --section="__DATA,__data" %t/sub-nosub | FileCheck %s --check-prefix SUB-NOSUB

## This test case demonstrates a shortcoming of LLD: If .subsections_via_symbols
## isn't enabled, we don't elide the contents of coalesced weak symbols if they
## are part of a section that has other non-coalesced symbols. In contrast, LD64
## does elide the contents.
# SUB-NOSUB:       Contents of (__DATA,__data) section
# SUB-NOSUB-NEXT:  0000000000001000    23 01 00 00 00 00 00 00 00 10 00 00 00 00 00 00
# SUB-NOSUB-NEXT:  0000000000001010    00 00 00 00 00 00 00 00 23 01 00 00 00 00 00 00
# SUB-NOSUB-EMPTY:
# SUB-NOSUB-NEXT:  SYMBOL TABLE:
# SUB-NOSUB-NEXT:  0000000000001000 l     O __DATA,__data _alias
# SUB-NOSUB-NEXT:  0000000000001008 l     O __DATA,__data _ref
# SUB-NOSUB-NEXT:  0000000000001010 l     O __DATA,__data _zeros
# SUB-NOSUB-NEXT:  0000000000001000 l     O __DATA,__data _alias
# SUB-NOSUB-NEXT:  0000000000001000  w    O __DATA,__data _weak
# SUB-NOSUB-NEXT:  0000000000000000         *UND* dyld_stub_binder

# RUN: %lld -lSystem -dylib %t/no-subsections.o %t/local-then-weak.o -o %t/nosub-sub
# RUN: llvm-objdump --macho --syms --section="__DATA,__data" %t/nosub-sub | FileCheck %s --check-prefix NOSUB-SUB

# NOSUB-SUB:       Contents of (__DATA,__data) section
# NOSUB-SUB-NEXT:  0000000000001000    00 00 00 00 00 00 00 00 23 01 00 00 00 00 00 00
# NOSUB-SUB-NEXT:  0000000000001010    08 10 00 00 00 00 00 00 {{$}}
# NOSUB-SUB-EMPTY:
# NOSUB-SUB-NEXT:  SYMBOL TABLE:
# NOSUB-SUB-NEXT:  0000000000001000 l     O __DATA,__data _zeros
# NOSUB-SUB-NEXT:  0000000000001008 l     O __DATA,__data _alias
# NOSUB-SUB-NEXT:  0000000000001008 l     O __DATA,__data _alias
# NOSUB-SUB-NEXT:  0000000000001010 l     O __DATA,__data _ref
# NOSUB-SUB-NEXT:  0000000000001008  w    O __DATA,__data _weak
# NOSUB-SUB-NEXT:  0000000000000000         *UND* dyld_stub_binder

## Verify that we don't drop any flags that the aliases have (such as
## .no_dead_strip). This is a regression test. We previously had subsections
## that were mistakenly stripped.

# RUN: %lld -lSystem -dead_strip %t/no-dead-strip.o -o %t/no-dead-strip
# RUN: llvm-objdump --macho --section-headers %t/no-dead-strip | FileCheck %s \
# RUN:   --check-prefix=NO-DEAD-STRIP
# NO-DEAD-STRIP: __data        00000010

#--- weak-then-local.s
.globl _weak
.weak_definition _weak
.data
_weak:
_alias:
  .quad 0x123

_ref:
  .quad _alias

.subsections_via_symbols

#--- local-then-weak.s
.globl _weak
.weak_definition _weak
.data
_alias:
_weak:
  .quad 0x123

_ref:
  .quad _alias

.subsections_via_symbols

#--- no-subsections.s
.globl _weak
.weak_definition _weak
.data
_zeros:
.space 8

_weak:
_alias:
  .quad 0x123

#--- no-dead-strip.s
.globl _main

_main:
  ret

.data
.no_dead_strip l_foo, l_bar

_foo:
l_foo:
  .quad 0x123

l_bar:
_bar:
  .quad 0x123

.subsections_via_symbols
