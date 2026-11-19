# REQUIRES: aarch64, x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/unaligned.s -o %t/unaligned.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/unaligned.s -o %t/unaligned-x86_64.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/four.s -o %t/four.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/four.s -o %t/four-x86_64.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/aligned.s -o %t/aligned.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/straddle.s -o %t/straddle.o

## Every chained fixup must be aligned, including the first one on a page, which
## has no predecessor to measure its distance from: to 8 bytes on arm64, and to
## DYLD_CHAINED_PTR_64's 4-byte stride on x86_64.
# RUN: not %lld -arch arm64 -dylib -fixup_chains %t/unaligned.o -o /dev/null 2>&1 \
# RUN:     | FileCheck %s -DALIGN=8 --implicit-check-not=error:
# RUN: not %lld -dylib -fixup_chains %t/unaligned-x86_64.o -o /dev/null 2>&1 \
# RUN:     | FileCheck %s -DALIGN=4 --implicit-check-not=error:
# CHECK-DAG: error: {{.*}}.o:(symbol _first+0x0): fixup is not [[ALIGN]]-byte aligned; stop packing structs containing pointers or link with -no_fixup_chains
# CHECK-DAG: error: {{.*}}.o:(symbol _shifted+0x0): fixup is not [[ALIGN]]-byte aligned; stop packing structs containing pointers or link with -no_fixup_chains
# CHECK-DAG: error: {{.*}}.o:(symbol _after_aligned+0x0): fixup is not [[ALIGN]]-byte aligned; stop packing structs containing pointers or link with -no_fixup_chains

## Like ld64, that includes arm64 pointers that the 4-byte stride could chain.
# RUN: not %lld -arch arm64 -dylib -fixup_chains -U _ext %t/four.o -o /dev/null 2>&1 \
# RUN:     | FileCheck %s --check-prefix=FOUR --implicit-check-not=error:
# FOUR-DAG: error: {{.*}}four.o:(symbol _p+0x0): fixup is not 8-byte aligned; stop packing structs containing pointers or link with -no_fixup_chains
# FOUR-DAG: error: {{.*}}four.o:(symbol _q+0x0): fixup is not 8-byte aligned; stop packing structs containing pointers or link with -no_fixup_chains

## ...but they work without chained fixups.
# RUN: %lld -arch arm64 -dylib -no_fixup_chains %t/unaligned.o -o /dev/null
# RUN: %lld -arch arm64 -dylib -no_fixup_chains -U _ext %t/four.o -o /dev/null

## A fixup can both straddle a page boundary and be unaligned; report both.
# RUN: not %lld -arch arm64 -dylib -fixup_chains %t/straddle.o -o /dev/null 2>&1 \
# RUN:     | FileCheck %s --check-prefix=STRADDLE --implicit-check-not=error:
# STRADDLE:      error: {{.*}}straddle.o:(symbol _p+0x0): fixup straddles page boundary; stop packing structs containing pointers or link with -no_fixup_chains
# STRADDLE-NEXT: error: {{.*}}straddle.o:(symbol _p+0x0): fixup is not 8-byte aligned; stop packing structs containing pointers or link with -no_fixup_chains

## 8-byte alignment is enough on arm64.
# RUN: %lld -arch arm64 -dylib -fixup_chains %t/aligned.o -o /dev/null

## On x86_64, ld64 falls back to opcodes for such pointers. We chain them
## instead, so 4-byte alignment is enough there, for rebases, binds, and the
## "next" fields that chain them.
# RUN: %lld -dylib -fixup_chains -U _ext %t/four-x86_64.o -o %t/four-x86_64
# RUN: llvm-objdump --macho --dyld-info --chained-fixups --syms %t/four-x86_64 \
# RUN:     | FileCheck %s --check-prefix=X86-FOUR
# X86-FOUR:      __DATA __data 0x[[#%X,P:]]  {{.*}} rebase 0x[[#%X,TARGET:]]
# X86-FOUR-NEXT: __DATA __data 0x[[#%X,P+8]] {{.*}} bind 0x0 flat-namespace _ext
# X86-FOUR:      page_start[0] = 4
# X86-FOUR:      SYMBOL TABLE:
# X86-FOUR-DAG:  [[#%.16x,P]] l O __DATA,__data _p
# X86-FOUR-DAG:  [[#%.16x,TARGET]] l F __TEXT,__text _target

#--- unaligned.s
.data
.p2align 3
.byte 0
## The first fixup on the page.
_first:
  .quad _target
## A multiple of 8 bytes after an unaligned fixup.
_shifted:
  .quad _target
.p2align 3
.quad _target
.byte 0
## Not a multiple of 8 bytes after an aligned fixup.
_after_aligned:
  .quad _target

.text
_target:
  ret

#--- four.s
.data
.p2align 3
.long 0
_p:
  .quad _target
_q:
  .quad _ext

.text
_target:
  ret

#--- aligned.s
.data
.p2align 3
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
