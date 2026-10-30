# REQUIRES: aarch64
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/unaligned.s -o %t/unaligned.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/aligned.s -o %t/aligned.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/straddle.s -o %t/straddle.o

## Every chained fixup must be 4-byte aligned, including the first one on a
## page, which has no predecessor to measure its distance from.
# RUN: not %lld -arch arm64 -dylib -fixup_chains %t/unaligned.o -o /dev/null 2>&1 \
# RUN:     | FileCheck %s --implicit-check-not=error:
# CHECK-DAG: error: {{.*}}unaligned.o:(symbol _first+0x0): fixup is not 4-byte aligned; stop packing structs containing pointers or link with -no_fixup_chains
# CHECK-DAG: error: {{.*}}unaligned.o:(symbol _shifted+0x0): fixup is not 4-byte aligned; stop packing structs containing pointers or link with -no_fixup_chains
# CHECK-DAG: error: {{.*}}unaligned.o:(symbol _after_aligned+0x0): fixup is not 4-byte aligned; stop packing structs containing pointers or link with -no_fixup_chains

## ...but they work without chained fixups.
# RUN: %lld -arch arm64 -dylib -no_fixup_chains %t/unaligned.o -o /dev/null

## A fixup can both straddle a page boundary and be unaligned; report both.
# RUN: not %lld -arch arm64 -dylib -fixup_chains %t/straddle.o -o /dev/null 2>&1 \
# RUN:     | FileCheck %s --check-prefix=STRADDLE --implicit-check-not=error:
# STRADDLE:      error: {{.*}}straddle.o:(symbol _p+0x0): fixup straddles page boundary; stop packing structs containing pointers or link with -no_fixup_chains
# STRADDLE-NEXT: error: {{.*}}straddle.o:(symbol _p+0x0): fixup is not 4-byte aligned; stop packing structs containing pointers or link with -no_fixup_chains

## 4-byte alignment is enough.
# RUN: %lld -arch arm64 -dylib -fixup_chains %t/aligned.o -o /dev/null

#--- unaligned.s
.data
.p2align 3
.byte 0
## The first fixup on the page.
_first:
  .quad _target
## A multiple of 4 bytes after an unaligned fixup.
_shifted:
  .quad _target
.p2align 3
.quad _target
.byte 0
## Not a multiple of 4 bytes after an aligned fixup.
_after_aligned:
  .quad _target

.text
_target:
  ret

#--- aligned.s
.data
.p2align 3
.long 0
_p:
  .quad _target

.text
_target:
  ret

#--- straddle.s
.data
.p2align 14
.space 0x3ffd
_p:
  .quad _target

.text
_target:
  ret
