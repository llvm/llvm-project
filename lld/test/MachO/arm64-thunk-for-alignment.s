# REQUIRES: aarch64
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/foo.s -o %t/foo.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/bar.s -o %t/bar.o
# RUN: %lld -dylib -arch arm64 -lSystem -o %t/out %t/foo.o %t/bar.o

# RUN: llvm-objdump --macho --syms %t/out | FileCheck %s
# CHECK: _bar.thunk.0

## Regression test for PR59259. Previously, we neglected to check section
## alignments when deciding when to create thunks.

## If we ignore alignment, the total size of _spacer1 + _spacer2 below is just
## under the limit at which we attempt to insert thunks between the spacers.
## However, with alignment accounted for, their total size ends up being
## 0x8000000, which is just above the max forward branch range, making thunk
## insertion necessary. Thus, not accounting for alignment led to an error.

#--- foo.s

_foo:
  b _bar

## Size of a `b` instruction.
.equ callSize, 4
## Refer to `slop` in TextOutputSection::finalize().
.equ slopSize, 12 * 256

_spacer1:
  .space 0x4000000 - slopSize - 2 * callSize - 1

.subsections_via_symbols

#--- bar.s
.globl _bar

.p2align 14
_spacer2:
  .space 0x4000000

_bar:
  ret

.subsections_via_symbols
