# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t.o
# RUN: %lld -arch arm64 -lSystem -o %t.out %t.o

## Regression test for PR51578.

.subsections_via_symbols

.globl _f1, _f2, _f3, _f4, _f5, _f6
.p2align 2
_f1: b _fn1
_f2: b _fn2
_f3: b _fn3
_f4: b _fn4
_f5: b _fn5
_f6: b _fn6
## 6 * 4 = 24 bytes for branches
## Currently leaves 12 bytes for one thunk, so 36 bytes.
## Uses < instead of <=, so 40 bytes.

.global _spacer1, _spacer2
## 0x8000000 is 128 MiB, one more than the forward branch limit,
## distributed over two functions since our thunk insertion algorithm
## can't deal with a single function that's 128 MiB.
## We leave just enough room so that the old thunking algorithm finalized
## both spacers when processing _f1 (24 bytes for the 4 bytes code for each
## of the 6 _f functions, 12 bytes for one thunk, 4 bytes because the forward
## branch range is 128 Mib - 4 bytes, and another 4 bytes because the algorithm
## uses `<` instead of `<=`, for a total of 44 bytes slop.) Of the slop, 20
## bytes are actually room for thunks.
## _fn1-_fn6 aren't finalized because then there wouldn't be room for a thunk.
## But when a thunk is inserted to jump from _f1 to _fn1, that needs 12 bytes
## but _f2 is only 4 bytes later, so after _f1 there are only
## 20-(12-4) = 12 bytes left, after _f2 only 12-(12-4) 4 bytes, and after
## _f3 there's no more room for thunks and we can't make progress.
## The fix is to leave room for many more thunks.
## The same construction as this test case can defeat that too with enough
## consecutive jumps, but in practice there aren't hundreds of consecutive
## jump instructions.

_spacer1:
.space 0x4000000
_spacer2:
.space 0x4000000 - 44

.globl _fn1, _fn2, _fn3, _fn4, _fn5, _fn6
.p2align 2
_fn1: ret
_fn2: ret
_fn3: ret
_fn4: ret
_fn5: ret
_fn6: ret

.globl _main
_main:
  ret
