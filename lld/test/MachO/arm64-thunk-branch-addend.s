# REQUIRES: aarch64

# RUN: rm -rf %t; mkdir %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t/input.o
# RUN: %lld -arch arm64 -lSystem -o %t/out %t/input.o
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t/out | FileCheck %s

## When two call sites branch to the same target symbol with different addends
## and each needs a branch-extension thunk, the thunks must be distinct.

.subsections_via_symbols

.text

## _target has an interior entry point at +8. A distant caller branches
## to both entry points, forcing a branch-extension thunk for each.
.globl _target
.p2align 2
_target:
  mov w0, #1
  ret
  mov w0, #2
  ret

## Spacer to push _main out of branch range of _target.
.globl _spacer
.p2align 2
_spacer:
  .space 0x8000000
  ret

.globl _main
.p2align 2
_main:
  bl _target
  bl _target+8
  ret

# CHECK-LABEL: <_main>:
# CHECK-NEXT:   bl 0x{{[0-9a-f]+}} <_target.thunk.0>
# CHECK-NEXT:   bl 0x{{[0-9a-f]+}} <_target+8.thunk.0>
# CHECK-NEXT:   ret

## The zero-addend thunk loads the page offset of _target.
# CHECK-LABEL: <_target.thunk.0>:
# CHECK-NEXT:   adrp  x16, 0x[[#%x, PAGE:]]
# CHECK-NEXT:   add   x16, x16, #[[#%u, OFF0:]]
# CHECK-NEXT:   br    x16

## The +8 thunk lives on the same page but loads an offset exactly 8
## bytes higher, so that `br x16` lands on `_target+8`.
# CHECK-LABEL: <_target+8.thunk.0>:
# CHECK-NEXT:   adrp  x16, 0x[[#PAGE]]
# CHECK-NEXT:   add   x16, x16, #[[#OFF0 + 8]]
# CHECK-NEXT:   br    x16
