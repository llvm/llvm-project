# REQUIRES: aarch64
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/foo.s -o %t/foo.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/bar.s -o %t/bar.o
# RUN: %lld -dylib -arch arm64 -lSystem -o %t/out %t/foo.o %t/bar.o

# RUN: llvm-objdump --macho --syms %t/out | FileCheck %s
# CHECK: _bar.thunk.0

## Regression test for PR59259. Previously, we neglected to check section
## alignments when deciding when to create thunks.

## If we ignore alignment, _bar is 0x7fff3fb bytes after _foo, just within the
## max forward branch range. However, aligning _spacer2's section adds 0xc05
## bytes of padding, placing _bar 0x8000000 bytes after _foo. This requires a
## thunk, and previously not accounting for that alignment led to an error.

#--- foo.s

.p2align 2
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
