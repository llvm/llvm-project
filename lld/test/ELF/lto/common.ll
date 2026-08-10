; REQUIRES: x86

;; Show that common symbols defined in both native objects and bitcode files are
;; properly resolved.

; RUN: rm -rf %t.dir
; RUN: split-file %s %t.dir
; RUN: cd %t.dir

;; Case 1: bitcode file has smaller symbol but larger alignment.
; RUN: llvm-as smaller-sym.ll -o smaller-sym.bc
; RUN: llc -filetype=obj larger-sym.ll -o larger-sym.o
; RUN: ld.lld smaller-sym.bc larger-sym.o -o 1.so -shared
; RUN: llvm-readobj -S --symbols 1.so | FileCheck %s -DALIGN=8

;; Case 2: bitcode file has larger symbol but smaller alignment.
; RUN: llvm-as larger-sym.ll -o larger-sym.bc
; RUN: llc -filetype=obj smaller-sym.ll -o smaller-sym.o
; RUN: ld.lld smaller-sym.o larger-sym.bc -o 2.so -shared
;; FIXME: This alignment should be 8, but LLD is ignoring the alignment of a
;; symbol in a native object file when the larger symbol is in a bitcode file.
;; See https://bugs.llvm.org/show_bug.cgi?id=47819.
; RUN: llvm-readobj -S --symbols 2.so | FileCheck %s -DALIGN=4

; CHECK:      Name: .bss
; CHECK-NEXT: Type: SHT_NOBITS
; CHECK-NEXT: Flags [
; CHECK-NEXT:   SHF_ALLOC
; CHECK-NEXT:   SHF_WRITE
; CHECK-NEXT: ]
; CHECK-NEXT: Address:
; CHECK-NEXT: Offset:
; CHECK-NEXT: Size: 2
; CHECK-NEXT: Link: 0
; CHECK-NEXT: Info: 0
; CHECK-NEXT: AddressAlignment: [[ALIGN]]

; CHECK:      Name: a
; CHECK-NEXT: Value:
; CHECK-NEXT: Size: 2
; CHECK-NEXT: Binding: Global
; CHECK-NEXT: Type: Object
; CHECK-NEXT: Other: 0
; CHECK-NEXT: Section: .bss

;--- smaller-sym.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = common global i8 0, align 8

;--- larger-sym.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = common global i16 0, align 4
