# REQUIRES: aarch64
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/first.s -o %t/first.o
# RUN: not %lld -arch arm64 -dylib -fixup_chains %t/first.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=FIRST
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/later.s -o %t/later.o
# RUN: not %lld -arch arm64 -dylib -fixup_chains %t/later.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=LATER
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/aligned.s -o %t/aligned.o
# RUN: %lld -arch arm64 -dylib -fixup_chains %t/aligned.o -o /dev/null

## The first fixup on a page has no predecessor to measure from, so check its
## alignment directly, including on a later page.
# FIRST: error: __DATA,__data, offset 1: fixup is unaligned (its address is not a multiple of the stride). Re-link with -no_fixup_chains
# LATER: error: __DATA,__data, offset 16385: fixup is unaligned (its address is not a multiple of the stride). Re-link with -no_fixup_chains

#--- first.s
.data
.p2align 3
.byte 0
.quad _target

.text
_target:
  ret

#--- later.s
.data
.p2align 14
.quad _target
.space 16377
.quad _target

.text
_target:
  ret

#--- aligned.s
.data
.p2align 14
.quad _target
.space 16376
.quad _target

.text
_target:
  ret
