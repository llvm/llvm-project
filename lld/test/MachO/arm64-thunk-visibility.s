# REQUIRES: aarch64

# foo.s and bar.s both contain TU-local symbols (think static function)
# with the same name, and both need a thunk..  This tests that ld64.lld doesn't
# create a duplicate symbol for the two functions.

# Test this both when the TU-local symbol is the branch source or target,
# and for both forward and backwards jumps.

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/foo.s -o %t/foo.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/bar.s -o %t/bar.o
# RUN: %lld -arch arm64 -lSystem -o %t.out %t/foo.o %t/bar.o

#--- foo.s

.subsections_via_symbols

# Note: No .globl, since these are TU-local symbols.
.p2align 2
_early_jumping_local_fn: b _some_late_external
.p2align 2
_early_landing_local_fn: ret

.globl _some_early_external
.p2align 2
_some_early_external: b _late_landing_local_fn

## 0x8000000 is 128 MiB, one more than the forward branch limit.
## Distribute that over two functions since our thunk insertion algorithm
## can't deal with a single function that's 128 MiB.
.global _spacer1, _spacer2
_spacer1:
.space 0x4000000
_spacer2:
.space 0x4000000

# Note: No .globl, since these are TU-local symbols.
.p2align 2
_late_jumping_local_fn: b _some_early_external
.p2align 2
_late_landing_local_fn: ret

.globl _some_late_external
.p2align 2
_some_late_external: b _early_landing_local_fn

#--- bar.s

.subsections_via_symbols

# Note: No .globl, since these are TU-local symbols.
.p2align 2
_early_jumping_local_fn: b _some_other_late_external
.p2align 2
_early_landing_local_fn: ret

.globl _some_other_early_external
.p2align 2
_some_other_early_external: b _late_landing_local_fn

## 0x8000000 is 128 MiB, one more than the forward branch limit.
## Distribute that over two functions since our thunk insertion algorithm
## can't deal with a single function that's 128 MiB.
.global _other_spacer1, _other_spacer1
_spacer1:
.space 0x4000000
_spacer2:
.space 0x4000000

# Note: No .globl, since these are TU-local symbols.
.p2align 2
_late_jumping_local_fn: b _some_other_early_external
.p2align 2
_late_landing_local_fn: ret

.globl _some_other_late_external
.p2align 2
_some_other_late_external: b _early_landing_local_fn

.globl _main
_main:
  ret
