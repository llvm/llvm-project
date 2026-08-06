# REQUIRES: aarch64

# RUN: rm -rf %t; mkdir %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t/input.o

## When two call sites branch to the same target symbol with different addends
## and each needs a branch-extension thunk, the thunks must be distinct.
# RUN: %lld -arch arm64 -lSystem -o %t/base %t/input.o
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t/base \
# RUN:   | FileCheck %s --check-prefixes=CHECK,BASE

## Regression test for the narrow corner case where the referent of a
## branch relocation is a Defined symbol that *also* has a __stubs entry
## (here forced via -interposable, which creates a stub for every extern
## Defined so callers can be interposed at runtime).
# RUN: %lld -arch arm64 -lSystem -interposable -o %t/interp %t/input.o
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t/interp \
# RUN:   | FileCheck %s --check-prefixes=CHECK,INTERP

.subsections_via_symbols

.text

## _target will become an interposable extern Defined under `-interposable`: it
## has a real body *and* a __stubs entry.
.globl _target
## Align it to a 4K boundary to make the test a little bit easier to write.
.p2align 12
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

## Capture _target's full VA.
# CHECK-LABEL: <_target>:
# CHECK-NEXT:   [[#%x, TARGET_VA:]]: mov w0, #1

# BASE-LABEL: <_main>:
# BASE-NEXT:    bl 0x{{[0-9a-f]+}} <_target.thunk.0>
# BASE-NEXT:    bl 0x{{[0-9a-f]+}} <_target+8.thunk.0>
# BASE-NEXT:    ret

## For -interposable: The addend=0 call takes the __stubs fast-path, so its bl
## target is an address inside the __stubs section. The addend=8 call must
## resolve against the local body.
# INTERP-LABEL: <_main>:
# INTERP-NEXT:  bl 0x[[#%x, STUB_CALL:]]
# INTERP-NEXT:  bl 0x{{[0-9a-f]+}} <_target+8.thunk.0>
# INTERP-NEXT:  ret

## The +0 thunk should be only present in BASE test.
# BASE-LABEL: <_target.thunk.0>:
# BASE-NEXT:    adrp  x16, 0x[[#TARGET_VA]]
# BASE-NEXT:    add   x16, x16, #0
# BASE-NEXT:    br    x16

## The +8 thunk must load _target+8 into x16: adrp loads _target (which is
## 4K-aligned), add supplies the +8.
# CHECK-LABEL: <_target+8.thunk.0>:
# CHECK-NEXT:   adrp  x16, 0x[[#TARGET_VA]]
# CHECK-NEXT:   add   x16, x16, #8
# CHECK-NEXT:   br    x16

# INTERP-LABEL: <__stubs>:
# INTERP:       [[#STUB_CALL]]: adrp x16,
