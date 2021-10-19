; REQUIRES: x86
;; Show that resolution of weak + global symbols works correctly when one is 
;; defined in native object and the other in a bitcode file.
;; The global symbol in both cases should be kept. LTO should throw away the
;; data for the discarded weak symbol defined in bitcode. The data for the
;; weak symbol in a native object will be kept, but will be unlabelled.

; RUN: rm -rf %t.dir
; RUN: split-file %s %t.dir
; RUN: llvm-as %t.dir/1.ll -o %t1.o
; RUN: llvm-mc -triple=x86_64-pc-linux %t.dir/2.s -o %t2.o -filetype=obj
; RUN: ld.lld %t1.o %t2.o -o %t.so -shared
; RUN: llvm-readobj --symbols -S --section-data %t.so | FileCheck %s

; CHECK:      Name: .data
; CHECK-NEXT: Type: SHT_PROGBITS
; CHECK-NEXT: Flags [
; CHECK-NEXT:   SHF_ALLOC
; CHECK-NEXT:   SHF_WRITE
; CHECK-NEXT: ]
; CHECK-NEXT: Address: 0x[[#%x,ADDR:]]
; CHECK-NEXT: Offset:
; CHECK-NEXT: Size: 12
; CHECK-NEXT: Link: 
; CHECK-NEXT: Info:
; CHECK-NEXT: AddressAlignment:
; CHECK-NEXT: EntrySize:
; CHECK-NEXT: SectionData (
; CHECK-NEXT:   0000: 09000000 05000000 04000000 |{{.*}}|
; CHECK-NEXT: )

; CHECK:      Name: a{{ }}
; CHECK-NEXT: Value: 0x[[#%x,ADDR]]

; CHECK:      Name: b{{ }}
; CHECK-NEXT: Value: 0x[[#%x,ADDR+8]]

;--- 1.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = weak global i32 8
@b = global i32 4

;--- 2.s
.data
.global a
a:
.long 9

.weak b
b:
.long 5
