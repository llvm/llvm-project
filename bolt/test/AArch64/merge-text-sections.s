## Test option `--merge-text-sections`: all the code emitted by BOLT is
## described by a single .text section header, with a local marker symbol
## recording the name and start address each folded section had before
## the merge.

# REQUIRES: system-linux

## The .cfi_* directives give the input an .eh_frame, which BOLT lays out
## immediately after the new code, so the merged section is followed by an
## unrelated allocatable section.
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe -q --entry=_start
# RUN: link_fdata --no-lbr %s %t.exe %t.fdata

## Baseline: without the option, hot and cold code get separate headers.
# RUN: llvm-bolt %t.exe -o %t.base --data %t.fdata --split-functions
# RUN: llvm-readelf -S %t.base | FileCheck %s --check-prefix=CHECK-BASE

# CHECK-BASE-DAG: ] .text {{.*}} AX
# CHECK-BASE-DAG: ] .text.cold {{.*}} AX

## Merged: one .text header, and no .text.cold.
# RUN: llvm-bolt %t.exe -o %t.merged --data %t.fdata --split-functions \
# RUN:         --merge-text-sections 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-INFO
# RUN: llvm-readelf -S %t.merged | FileCheck %s --check-prefix=CHECK-SEC
# RUN: llvm-readelf -S %t.merged | FileCheck %s --check-prefix=CHECK-NOCOLD

# CHECK-INFO: BOLT-INFO: merged {{[0-9]+}} code sections into .text

# CHECK-SEC: ] .text {{.*}} AX
## The unwind table sits immediately after the merged code and
## must still be described by its own header.
# CHECK-SEC: ] .eh_frame {{.*}} A
# CHECK-NOCOLD-NOT: .text.cold

## The original .text is preserved as .bolt.org.text. It lives at its input
## address, is not adjacent to the new code, and is never merged.
# RUN: llvm-readelf -S %t.merged | FileCheck %s --check-prefix=CHECK-ORG
# CHECK-ORG: ] .bolt.org.text {{.*}} AX

## Marker symbols are local with size zero and resolve to the merged .text,
## as does the cold fragment of the split function, which would otherwise be
## left pointing at SHN_UNDEF.
# RUN: llvm-readelf -s %t.merged | grep -E "bolt\.pre_merge|chain\.cold\." \
# RUN:   | sort -k8 > %t.merged.markers
# RUN: FileCheck %s --check-prefix=CHECK-SYMS --input-file=%t.merged.markers
# RUN: FileCheck %s --check-prefix=CHECK-NOUND --input-file=%t.merged.markers

# CHECK-SYMS:      {{[0-9]+}}: {{[0-9a-f]+}} 0 NOTYPE LOCAL DEFAULT
# CHECK-SYMS-SAME: [[TEXT:[0-9]+]] .bolt.pre_merge.text{{$}}
# CHECK-SYMS:      {{[0-9]+}}: {{[0-9a-f]+}} 0 NOTYPE LOCAL DEFAULT
# CHECK-SYMS-SAME: [[TEXT]] .bolt.pre_merge.text.cold{{$}}
# CHECK-SYMS:      FUNC LOCAL DEFAULT [[TEXT]] chain.cold.0
# CHECK-NOUND-NOT: UND

## Merging only rewrites section headers, so every function and fragment keeps
## the address and size it had without the option.
# RUN: llvm-readelf -s %t.base | grep " FUNC " | awk '{print $2, $3, $8}' \
# RUN:   | sort > %t.base.syms
# RUN: llvm-readelf -s %t.merged | grep " FUNC " | awk '{print $2, $3, $8}' \
# RUN:   | sort > %t.merged.syms
# RUN: diff %t.base.syms %t.merged.syms

## --hot-functions-at-end lays cold code first. Merging still applies: the run
## of new code is contiguous either way, and the marker preserves the identity
## of the head section even though it is renamed to .text.
# RUN: llvm-bolt %t.exe -o %t.rev --data %t.fdata --split-functions \
# RUN:         --hot-functions-at-end --merge-text-sections
# RUN: llvm-readelf -S %t.rev | FileCheck %s --check-prefix=CHECK-SEC
# RUN: llvm-readelf -S %t.rev | FileCheck %s --check-prefix=CHECK-NOCOLD
# RUN: llvm-readelf -s %t.rev | grep -E "bolt\.pre_merge|chain\.cold\." \
# RUN:   | sort -k8 > %t.rev.markers
# RUN: FileCheck %s --check-prefix=CHECK-SYMS --input-file=%t.rev.markers
# RUN: FileCheck %s --check-prefix=CHECK-NOUND --input-file=%t.rev.markers

## The option requires relocation mode.
# RUN: ld.lld %t.o -o %t.norelocs --entry=_start
# RUN: not llvm-bolt %t.norelocs -o %t.null --merge-text-sections 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-NORELOC

# CHECK-NORELOC: BOLT-ERROR: --merge-text-sections requires relocation mode

## Verify `--merge-text-sections` works with `--use-old-text`.
# RUN: llvm-bolt %t.exe -o %t.uot --data %t.fdata --split-functions \
# RUN:         --use-old-text --merge-text-sections --align-text=4
# RUN: llvm-readelf -S %t.uot | FileCheck %s --check-prefix=CHECK-SEC
# RUN: llvm-readelf -S %t.uot | FileCheck %s --check-prefix=CHECK-NOCOLD
# RUN: llvm-readelf -S %t.uot | FileCheck %s --check-prefix=CHECK-ORG
# RUN: llvm-readelf -s %t.uot | grep -E "bolt\.pre_merge|chain\.cold\." \
# RUN:   | sort -k8 > %t.uot.markers
# RUN: FileCheck %s --check-prefix=CHECK-SYMS --input-file=%t.uot.markers
# RUN: FileCheck %s --check-prefix=CHECK-NOUND --input-file=%t.uot.markers

        .text
        .globl  _start
        .type   _start, %function
_start:
        .cfi_startproc
        stp     x29, x30, [sp, #-16]!
        .cfi_def_cfa_offset 16
        .cfi_offset w30, -8
        .cfi_offset w29, -16
        mov     x29, sp
        .cfi_def_cfa w29, 16
        mov     w0, #1
        bl      chain
        ldp     x29, x30, [sp], #16
        ret
        .cfi_endproc
        .size   _start, .-_start

        .globl  chain
        .type   chain, %function
chain:
.entry_bb:
# FDATA: 1 chain #.entry_bb# 100
        .cfi_startproc
        stp     x29, x30, [sp, #-16]!
        .cfi_def_cfa_offset 16
        .cfi_offset w30, -8
        .cfi_offset w29, -16
        mov     x29, sp
        .cfi_def_cfa w29, 16
        cmp     w0, #2
        b.ge    .Lcold_bb
        mov     w0, #5
        ldp     x29, x30, [sp], #16
        ret
.Lcold_bb:
        add     w0, w0, #1
        add     w0, w0, #1
        add     w0, w0, #1
        add     w0, w0, #1
        add     w0, w0, #1
        add     w0, w0, #1
        add     w0, w0, #1
        add     w0, w0, #1
        ldp     x29, x30, [sp], #16
        ret
        .cfi_endproc
        .size   chain, .-chain

## Filler so the original .text section has room for BOLT generated hot and
## cold sections under `--use-old-text`.
        .p2align 6
        .globl  filler
        .type   filler, %function
filler:
        .rept 32
        ret
        .endr
        .size filler, .-filler
