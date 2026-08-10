; REQUIRES: x86

;; Test weak symbols are supported in LTO. The first definition should be
;; used regardless of whether it is from a bitcode file or native object.

; RUN: split-file %s %t

; RUN: llvm-as %t/size1.ll -o %t-size1.bc
; RUN: llvm-as %t/size2.ll -o %t-size2.bc
; RUN: llc %t/size4.ll -o %t-size4.o -filetype=obj

; RUN: ld.lld %t-size1.bc %t-size2.bc -o %t.so -shared
; RUN: llvm-readobj --symbols %t.so | FileCheck %s -DSIZE=1

; RUN: ld.lld %t-size2.bc %t-size1.bc -o %t2.so -shared
; RUN: llvm-readobj --symbols %t2.so | FileCheck %s -DSIZE=2

; RUN: ld.lld %t-size1.bc %t-size4.o -o %t3.so -shared
; RUN: llvm-readobj --symbols %t3.so | FileCheck %s -DSIZE=1

; RUN: ld.lld %t-size4.o %t-size1.bc -o %t4.so -shared
; RUN: llvm-readobj --symbols %t4.so | FileCheck %s -DSIZE=4

; CHECK:      Name: a
; CHECK-NEXT: Value:
; CHECK-NEXT: Size: [[SIZE]]
; CHECK-NEXT: Binding: Weak
; CHECK-NEXT: Type: Object
; CHECK-NEXT: Other: 0
; CHECK-NEXT: Section: .data

;--- size1.ll
target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

@a = weak global i8 1

;--- size2.ll
target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

@a = weak global i16 1

;--- size4.ll
target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

@a = weak global i32 1
