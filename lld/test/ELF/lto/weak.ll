; REQUIRES: x86

;; Test weak symbols are supported in LTO.

; RUN: llvm-as %s -o %t.o
; RUN: ld.lld %t.o %t.o -o %t.so -shared
; RUN: llvm-readobj --symbols %t.so | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define weak void @f() {
  ret void
}

; CHECK:      Name: f
; CHECK-NEXT: Value:
; CHECK-NEXT: Size: 1
; CHECK-NEXT: Binding: Weak
; CHECK-NEXT: Type: Function
; CHECK-NEXT: Other: 0
; CHECK-NEXT: Section: .text
