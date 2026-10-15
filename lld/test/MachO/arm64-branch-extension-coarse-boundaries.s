# REQUIRES: aarch64

## Cover coarse-boundary preference for thunks and dense-boundary fallback
## for islands with separate input layouts.

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/thunk.s -o %t/thunk.o
## Invalid spacing must fail.
# RUN: not %lld -arch arm64 -dylib -o %t/invalid %t/thunk.o \
# RUN:   -arm64-branch-extension-coarse-boundary-spacing 4294967296 2>&1 \
# RUN:   | FileCheck %s --check-prefix=INVALID-SPACING
# INVALID-SPACING: error: -arm64-branch-extension-coarse-boundary-spacing: expected an integer in the range [0, 4294967295], but got '4294967296'

# RUN: %lld -arch arm64 -dylib -o %t/thunk %t/thunk.o \
# RUN:   -arm64-islands-max-hops 0
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t/thunk \
# RUN:   | FileCheck %s --check-prefix=THUNK
# RUN: %lld -arch arm64 -dylib -o %t/thunk-dense %t/thunk.o \
# RUN:   -arm64-islands-max-hops 0 -arm64-branch-extension-coarse-boundary-spacing 262144
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t/thunk-dense \
# RUN:   | FileCheck %s --check-prefix=THUNK-DENSE
# RUN: %lld -arch arm64 -dylib -o %t/thunk-all %t/thunk.o \
# RUN:   -arm64-islands-max-hops 0 -arm64-branch-extension-coarse-boundary-spacing 0
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t/thunk-all \
# RUN:   | FileCheck %s --check-prefix=THUNK-DENSE
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/island.s -o %t/island.o
# RUN: %lld -arch arm64 -dylib -o %t/island %t/island.o \
# RUN:   -arm64-islands-max-hops 4294967295 --verbose 2> %t/island.log
# RUN: FileCheck %s --check-prefix=ISLAND-LOG --input-file=%t/island.log
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t/island \
# RUN:   | FileCheck %s --check-prefix=ISLAND

## Prefer a coarse boundary over a denser boundary even when the latter is
## farther inward in the thunk's legal placement window. The coarse boundary
## follows _coarse; the boundary following _dense is only 256 KiB later and is
## therefore omitted from the 1 MiB-spaced coarse catalog.

## Input layout:
##
##   +0       +0x7000008         +0x7040008          +0x10040008
##   _main    end of _coarse     end of _dense      _target
##            |---- 256 KiB ----|
##            chosen boundary   also in caller's range
##
## Expected routing:
##
##   _main --> _target.thunk.0 ==> _target
##
## Both boundaries are below the forward BL limit +0x7fffffc. Prefer the
## coarse boundary; the dense boundary is omitted from the coarse catalog.
## The inserted thunk shifts later inputs.

# THUNK-LABEL: <_main>:
# THUNK-NEXT: bl 0x[[#%x, THUNK:]] <_target.thunk.0>
# THUNK-LABEL: <_coarse>:
# THUNK: [[#THUNK]] <_target.thunk.0>:
# THUNK-LABEL: <_dense>:
# THUNK-LABEL: <_gap>:
# THUNK-LABEL: <_target>:

## With 256 KiB spacing or zero, _dense is also a coarse candidate. The thunk
## moves after it, farther inward in the caller's legal placement window.
# THUNK-DENSE-LABEL: <_main>:
# THUNK-DENSE-NEXT: bl 0x[[#%x, DENSE_ADDR:]] <_target.thunk.0>
# THUNK-DENSE-LABEL: <_coarse>:
# THUNK-DENSE-LABEL: <_dense>:
# THUNK-DENSE: [[#DENSE_ADDR]] <_target.thunk.0>:
# THUNK-DENSE-LABEL: <_gap>:
# THUNK-DENSE-LABEL: <_target>:

## The coarse placement catalog has no legal island position in the target's
## narrow branch window. The planner must fall back to the dense real-input
## boundaries and select the boundary after _dense without adding a relay.

## Input layout:
##
##   +0       +0x7fbfffc         +0x7fffffc          +0xffffff0
##   _main    end of _coarse     end of _dense      _target
##            |---- 256 KiB ----|                  (before insertion)
##                            forward BL limit
##
## Expected routing:
##
##   _main --> _target.island.0 --> _target
##
## The island occupies the boundary after _dense at +0x7fffffc and moves
## _target to +0xffffff4. The coarse boundary cannot reach the target.

# ISLAND-LOG: branch extender for __TEXT: passes = 1
# ISLAND-LOG-SAME: total extenders = 1

# ISLAND-LABEL: <_main>:
# ISLAND-NEXT: bl 0x[[#%x, ISLAND:]] <_target.island.0>
# ISLAND-LABEL: <_coarse>:
# ISLAND-LABEL: <_dense>:
# ISLAND: [[#ISLAND]] <_target.island.0>:
# ISLAND-LABEL: <_tail>:
# ISLAND-LABEL: <_target>:

#--- thunk.s
.subsections_via_symbols
.text

.globl _main
_main:
  bl _target
  ret

.globl _coarse
_coarse:
  .space 0x7000000

.globl _dense
_dense:
  .space 0x40000

.globl _gap
_gap:
  .space 0x9000000

.globl _target
_target:
  ret

#--- island.s
.subsections_via_symbols
.text

.globl _main
.p2align 2
_main:
  bl _target
  ret

## ARM64's forward BRANCH26 range is 0x7fffffc. _coarse ends 256 KiB before
## the only legal boundary.
.globl _coarse
_coarse:
  .space 0x7fbfff4

.globl _dense
_dense:
  .space 0x40000

## Before insertion, the exact legal window is the final eight bytes of the
## call range. Inserting the island shifts _target by four bytes and leaves a
## final four-byte legal window containing only the _dense boundary.
.globl _tail
_tail:
  .space 0x7fffff4

.globl _target
_target:
  ret
