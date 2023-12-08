; REQUIRES: x86

;; NOTE: We deviate significantly from ld64's behavior here. We treat common
;; bitcode symbols like regular common symbols, but ld64 gives them different
;; (and IMO very strange) precedence. This test documents the differences.

; RUN: rm -rf %t; split-file %s %t
; RUN: opt -module-summary %t/test.ll -o %t/test.o
; RUN: opt -module-summary %t/same-size.ll -o %t/same-size.o
; RUN: opt -module-summary %t/smaller-size.ll -o %t/smaller-size.o
; RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/same-size.s -o %t/same-size-asm.o
; RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/smaller-size.s -o %t/smaller-size-asm.o

;; ld64: Common bitcode symbols all have equal precedence, regardless of size or
;; alignment.
;; lld: We pick the symbol with the larger size, regardless of alignment.
; RUN: %lld -dylib %t/test.o %t/smaller-size.o -order_file %t/order -o %t/test
; RUN: llvm-objdump --section-headers --syms %t/test | FileCheck %s -D#SIZE=2 -D#ALIGN=8
; RUN: %lld -dylib %t/smaller-size.o %t/test.o -order_file %t/order -o %t/test
; RUN: llvm-objdump --section-headers --syms %t/test | FileCheck %s -D#SIZE=2 -D#ALIGN=8
; COM (ld64): llvm-objdump --section-headers --syms %t/test | FileCheck %s -D#SIZE=1 -D#ALIGN=16

;; ld64: Common bitcode symbols all have equal precedence, regardless of size or
;; alignment.
;; lld: When the sizes are equal, we pick the symbol whose file occurs later in
;; the command-line argument list.
; RUN: %lld -dylib %t/test.o %t/same-size.o -order_file %t/order -o %t/test
; RUN: llvm-objdump --section-headers --syms %t/test | FileCheck %s -D#SIZE=2 -D#ALIGN=16
; COM (ld64): llvm-objdump --section-headers --syms %t/test | FileCheck %s -D#SIZE=2 -D#ALIGN=8
; RUN: %lld -dylib %t/same-size.o %t/test.o -order_file %t/order -o %t/test
; RUN: llvm-objdump --section-headers --syms %t/test | FileCheck %s -D#SIZE=2 -D#ALIGN=8

;; ld64: Non-bitcode common symbols take precedence.
;; lld: We pick the symbol with the larger size, regardless of alignment.
; RUN: %lld -dylib %t/test.o %t/smaller-size-asm.o -order_file %t/order -o %t/test
; RUN: llvm-objdump --section-headers --syms %t/test | FileCheck %s -D#SIZE=2 -D#ALIGN=8
; COM (ld64): llvm-objdump --section-headers --syms %t/test | FileCheck %s -D#SIZE=1 -D#ALIGN=16
; RUN: %lld -dylib %t/smaller-size-asm.o %t/test.o -order_file %t/order -o %t/test
; RUN: llvm-objdump --section-headers --syms %t/test | FileCheck %s -D#SIZE=2 -D#ALIGN=8
; COM (ld64): llvm-objdump --section-headers --syms %t/test | FileCheck %s -D#SIZE=1 -D#ALIGN=16

; RUN: %lld -dylib %t/test.o %t/same-size-asm.o -order_file %t/order -o %t/test
; RUN: llvm-objdump --section-headers --syms %t/test | FileCheck %s -D#SIZE=2 -D#ALIGN=16
; RUN: %lld -dylib %t/same-size-asm.o %t/test.o -order_file %t/order -o %t/test
; RUN: llvm-objdump --section-headers --syms %t/test | FileCheck %s -D#SIZE=2 -D#ALIGN=8
; COM (ld64): llvm-objdump --section-headers --syms %t/test | FileCheck %s -D#SIZE=2 -D#ALIGN=16

; CHECK-LABEL: Sections:
; CHECK:       __common      {{[0-9a-f]+}} [[#%x, COMMON_START:]]  BSS
;
; CHECK-LABEL: SYMBOL TABLE:
; CHECK-DAG:   [[#%.16x, COMMON_START]]         g     O __DATA,__common _check_size
; CHECK-DAG:   [[#%.16x, COMMON_START + SIZE]]  g     O __DATA,__common _end_marker
; CHECK-DAG:   [[#%.16x, COMMON_START + ALIGN]] g     O __DATA,__common _check_alignment

;--- order
;; Order is important as we determine the size of a given symbol via the
;; address of the next symbol.
_check_size
_end_marker
_check_alignment

;--- smaller-size.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

@check_size = common global i8 0, align 1
@check_alignment = common global i8 0, align 16

;--- same-size.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

@check_size = common global i16 0, align 1
@check_alignment = common global i16 0, align 16

;--- smaller-size.s
.comm _check_size, 1, 1
.comm _check_alignment, 1, 4

;--- same-size.s
.comm _check_size, 2, 1
.comm _check_alignment, 2, 4

;--- test.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"
@check_size = common global i16 0, align 1
@end_marker = common global i8 0
@check_alignment = common global i16 0, align 8
