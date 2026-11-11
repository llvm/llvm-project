# REQUIRES: aarch64

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos11 %t/data.s -o %t/data.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos11 %t/text.s -o %t/text.o
# RUN: %lld -dylib -arch arm64 --icf=all %t/data.o %t/text.o -o %t/out
# RUN: llvm-nm %t/out | FileCheck %s

## data.o does not use subsections-via-symbols, so _a, _b, _c and _d are all
## defined in one input section, at offsets 0, 8, 16 and 24. _f1 and _f2 have
## identical bytes and their relocations reference that same input section, but
## at different offsets: _f1 -> (_a, _d), _f2 -> (_b, _c). The referent offsets
## sum to the same value, so the two functions land in the same ICF hash
## bucket; the full comparison must tell them apart by the offsets of the
## referent symbols. _f3 is identical to _f1 and should still be folded.

## llvm-nm sorts symbols by name.
# CHECK:     [[#%.16x,F1:]] T _f1
# CHECK-NOT: [[#F1]] T _f2
# CHECK:     [[#F1]] T _f3

#--- data.s
.section __TEXT,__const
.globl _a, _b, _c, _d
.p2align 3
_a:
  .quad 1
_b:
  .quad 2
_c:
  .quad 3
_d:
  .quad 4

#--- text.s
.section __TEXT,__text,regular,pure_instructions
.globl _f1, _f2, _f3
.p2align 2
_f1:
  adrp x0, _a@PAGE
  add x0, x0, _a@PAGEOFF
  adrp x1, _d@PAGE
  add x1, x1, _d@PAGEOFF
  ret

_f2:
  adrp x0, _b@PAGE
  add x0, x0, _b@PAGEOFF
  adrp x1, _c@PAGE
  add x1, x1, _c@PAGEOFF
  ret

_f3:
  adrp x0, _a@PAGE
  add x0, x0, _a@PAGEOFF
  adrp x1, _d@PAGE
  add x1, x1, _d@PAGEOFF
  ret

.subsections_via_symbols
