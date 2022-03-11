# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/align-empty.s -o %t/align-empty.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/align-4-0.s -o %t/align-4-0.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/align-4-2.s -o %t/align-4-2.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/align-16-0.s -o %t/align-16-0.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/align-16-2.s -o %t/align-16-2.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/align-16-4.s -o %t/align-16-4.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/align-16-8.s -o %t/align-16-8.o

## Check that we preserve the alignment of cstrings. Alignment is determined
## not by section alignment but by the number of trailing zeros of the cstring's
## address in the input object file.

## The non-dedup case is not particularly interesting since the null bytes don't
## get dedup'ed, meaning that the output strings get their offsets "naturally"
## preserved.

# RUN: %lld -dylib %t/align-empty.o %t/align-4-0.o %t/align-16-0.o -o %t/align-4-0-16-0
# RUN: llvm-objdump --macho --section="__TEXT,__cstring" %t/align-4-0-16-0 | \
# RUN:   FileCheck %s -D#OFF1=4 -D#OFF2=16
# RUN: %lld -dylib %t/align-empty.o %t/align-16-0.o %t/align-4-0.o -o %t/align-16-0-4-0
# RUN: llvm-objdump --macho --section="__TEXT,__cstring" %t/align-16-0-4-0 | \
# RUN:   FileCheck %s -D#OFF1=16 -D#OFF2=20

# RUN: %lld -dylib %t/align-empty.o %t/align-4-2.o %t/align-16-0.o -o %t/align-4-2-16-0
# RUN: llvm-objdump --macho --section="__TEXT,__cstring" %t/align-4-2-16-0 | \
# RUN:   FileCheck %s -D#OFF1=6 -D#OFF2=16
# RUN: %lld -dylib %t/align-empty.o %t/align-16-0.o %t/align-4-2.o -o %t/align-16-0-4-2
# RUN: llvm-objdump --macho --section="__TEXT,__cstring" %t/align-16-0-4-2 | \
# RUN:   FileCheck %s -D#OFF1=16 -D#OFF2=22

# RUN: %lld -dylib %t/align-empty.o %t/align-4-0.o %t/align-16-2.o -o %t/align-4-0-16-2
# RUN: llvm-objdump --macho --section="__TEXT,__cstring" %t/align-4-0-16-2 | \
# RUN:   FileCheck %s -D#OFF1=4 -D#OFF2=18
# RUN: %lld -dylib %t/align-empty.o %t/align-16-2.o %t/align-4-0.o -o %t/align-16-2-4-0
# RUN: llvm-objdump --macho --section="__TEXT,__cstring" %t/align-16-2-4-0 | \
# RUN:   FileCheck %s -D#OFF1=18 -D#OFF2=20

# CHECK:       Contents of (__TEXT,__cstring) section
# CHECK-NEXT:  [[#%.16x,START:]]     {{$}}
# CHECK:       [[#%.16x,START+OFF1]] a{{$}}
# CHECK:       [[#%.16x,START+OFF2]] a{{$}}
# CHECK-EMPTY:

## The dedup cases are more interesting...

## Same offset, different alignments => pick higher alignment
# RUN: %lld -dylib --deduplicate-literals %t/align-empty.o %t/align-4-0.o %t/align-16-0.o -o %t/dedup-4-0-16-0
# RUN: llvm-objdump --macho --section="__TEXT,__cstring" %t/dedup-4-0-16-0 | \
# RUN:   FileCheck %s --check-prefix=DEDUP -D#OFF=16
# RUN: %lld -dylib --deduplicate-literals %t/align-empty.o %t/align-16-0.o %t/align-4-0.o -o %t/dedup-16-0-4-0
# RUN: llvm-objdump --macho --section="__TEXT,__cstring" %t/dedup-16-0-4-0 | \
# RUN:   FileCheck %s --check-prefix=DEDUP -D#OFF=16

## 16 byte alignment vs 2 byte offset => align to 16 bytes
# RUN: %lld -dylib --deduplicate-literals %t/align-empty.o %t/align-4-2.o %t/align-16-0.o -o %t/dedup-4-2-16-0
# RUN: llvm-objdump --macho --section="__TEXT,__cstring" %t/dedup-4-2-16-0 | \
# RUN:   FileCheck %s --check-prefix=DEDUP -D#OFF=16
# RUN: %lld -dylib --deduplicate-literals %t/align-empty.o %t/align-16-0.o %t/align-4-2.o -o %t/dedup-16-0-4-2
# RUN: llvm-objdump --macho --section="__TEXT,__cstring" %t/dedup-16-0-4-2 | \
# RUN:   FileCheck %s --check-prefix=DEDUP -D#OFF=16

## 4 byte alignment vs 2 byte offset => align to 4 bytes
# RUN: %lld -dylib --deduplicate-literals %t/align-empty.o %t/align-4-0.o %t/align-16-2.o -o %t/dedup-4-0-16-2
# RUN: llvm-objdump --macho --section="__TEXT,__cstring" %t/dedup-4-0-16-2 | \
# RUN:   FileCheck %s --check-prefix=DEDUP -D#OFF=4
# RUN: %lld -dylib --deduplicate-literals %t/align-empty.o %t/align-16-2.o %t/align-4-0.o -o %t/dedup-16-2-4-0
# RUN: llvm-objdump --macho --section="__TEXT,__cstring" %t/dedup-16-2-4-0 | \
# RUN:   FileCheck %s --check-prefix=DEDUP -D#OFF=4

## Both inputs are 4-byte aligned, one via offset and the other via section alignment
# RUN: %lld -dylib --deduplicate-literals %t/align-empty.o %t/align-4-0.o %t/align-16-4.o -o %t/dedup-4-0-16-4
# RUN: llvm-objdump --macho --section="__TEXT,__cstring" %t/dedup-4-0-16-4 | \
# RUN:   FileCheck %s --check-prefix=DEDUP -D#OFF=4
# RUN: %lld -dylib --deduplicate-literals %t/align-empty.o %t/align-16-4.o %t/align-4-0.o -o %t/dedup-16-4-4-0
# RUN: llvm-objdump --macho --section="__TEXT,__cstring" %t/dedup-16-4-4-0 | \
# RUN:   FileCheck %s --check-prefix=DEDUP -D#OFF=4

## 8-byte offset vs 4-byte section alignment => align to 8 bytes
# RUN: %lld -dylib --deduplicate-literals %t/align-empty.o %t/align-4-0.o %t/align-16-8.o -o %t/dedup-4-0-16-8
# RUN: llvm-objdump --macho --section="__TEXT,__cstring" %t/dedup-4-0-16-8 | \
# RUN:   FileCheck %s --check-prefix=DEDUP -D#OFF=8
# RUN: %lld -dylib --deduplicate-literals %t/align-empty.o %t/align-16-8.o %t/align-4-0.o -o %t/dedup-16-8-4-0
# RUN: llvm-objdump --macho --section="__TEXT,__cstring" %t/dedup-16-8-4-0 | \
# RUN:   FileCheck %s --check-prefix=DEDUP -D#OFF=8

# DEDUP:       Contents of (__TEXT,__cstring) section
# DEDUP-NEXT:  [[#%.16x,START:]]    {{$}}
# DEDUP:       [[#%.16x,START+OFF]] a{{$}}
# DEDUP-EMPTY:

#--- align-empty.s
## We use this file to create an empty string at the start of every output
## file's .cstring section. This makes the test cases more interesting since LLD
## can't place the string "a" at the trivially-aligned zero offset.
.cstring
.p2align 2
.asciz ""

#--- align-4-0.s
.cstring
.p2align 2
.asciz "a"

#--- align-4-2.s
.cstring
.p2align 2
.zero 0x2
.asciz "a"

#--- align-16-0.s
.cstring
.p2align 4
.asciz "a"

#--- align-16-2.s
.cstring
.p2align 4
.zero 0x2
.asciz "a"

#--- align-16-4.s
.cstring
.p2align 4
.zero 0x4
.asciz "a"

#--- align-16-8.s
.cstring
.p2align 4
.zero 0x8
.asciz "a"
