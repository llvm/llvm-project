# REQUIRES: x86
## This test checks that when we coalesce weak definitions, any private-label
## aliases to those weak defs don't cause the coalesced data to be retained.
## This test explicitly creates those private-label symbols, but it was actually
## motivated by MC's aarch64 backend which automatically creates them when
## emitting object files. I've chosen to explicitly create them here since we
## can then reference those symbols for a more complete test.
##
## Not retaining the data matters for more than just size -- we have a use case
## that depends on proper data coalescing to emit a valid file format.
##
## ld64 actually treats all local symbol aliases (not just the private ones) the
## same way. But implementing this is harder -- we would have to create those
## symbols first (so we can emit their names later), but we would have to
## ensure the linker correctly shuffles them around when their aliasees get
## coalesced. Emulating the behavior of weak binds for non-private symbols would
## be even trickier. Let's just deal with private-label symbols for now until we
## find a use case for more general local symbols.
##
## Finally, ld64 does all this regardless of whether .subsections_via_symbols is
## specified. We don't. But again, given how rare the lack of that directive is
## (I've only seen it from hand-written assembly inputs), I don't think we need
## to worry about it.

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/weak-then-private.s -o %t/weak-then-private.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/private-then-weak.s -o %t/private-then-weak.o
# RUN: %lld -dylib %t/weak-then-private.o %t/private-then-weak.o -o %t/test1
# RUN: %lld -dylib %t/private-then-weak.o %t/weak-then-private.o -o %t/test2
# RUN: llvm-objdump --macho --syms --section="__DATA,__data" --weak-bind %t/test1 | FileCheck %s
# RUN: llvm-objdump --macho --syms --section="__DATA,__data" --weak-bind %t/test2 | FileCheck %s

## Check that we only have one copy of 0x123 in the data, not two.
# CHECK:       Contents of (__DATA,__data) section
# CHECK-NEXT:  0000000000001000  23 01 00 00 00 00 00 00 00 10 00 00 00 00 00 00 {{$}}
# CHECK-NEXT:  0000000000001010  00 10 00 00 00 00 00 00 {{$}}
# CHECK-EMPTY:
# CHECK-NEXT:  SYMBOL TABLE:
# CHECK-NEXT:  0000000000001008 l     O __DATA,__data _ref
# CHECK-NEXT:  0000000000001010 l     O __DATA,__data _ref
# CHECK-NEXT:  0000000000001000  w    O __DATA,__data _weak
# CHECK-NEXT:  0000000000000000         *UND* dyld_stub_binder
# CHECK-EMPTY:
## Even though the references were to the non-weak `l_ignored` aliases, we
## should still emit weak binds as if they were the `_weak` symbol itself.
# CHECK-NEXT:  Weak bind table:
# CHECK-NEXT:  segment  section            address     type       addend   symbol
# CHECK-NEXT:  __DATA   __data             0x00001008 pointer         0   _weak
# CHECK-NEXT:  __DATA   __data             0x00001010 pointer         0   _weak

#--- weak-then-private.s
.globl _weak
.weak_definition _weak
.data
_weak:
l_ignored:
  .quad 0x123

_ref:
  .quad l_ignored

.subsections_via_symbols

#--- private-then-weak.s
.globl _weak
.weak_definition _weak
.data
l_ignored:
_weak:
  .quad 0x123

_ref:
  .quad l_ignored

.subsections_via_symbols
