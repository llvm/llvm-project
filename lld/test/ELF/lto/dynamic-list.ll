; REQUIRES: x86

;; Test we parse dynamic lists before LTO, so symbols can be exported correctly.

; RUN: llvm-as %s -o %t.o
; RUN: echo "{ foo; };" > %t.list
; RUN: ld.lld -o %t --dynamic-list %t.list -pie %t.o
; RUN: llvm-readobj --dyn-syms %t | FileCheck %s

; CHECK:      Name:     foo
; CHECK-NEXT: Value:
; CHECK-NEXT: Size:     1
; CHECK-NEXT: Binding:  Global (0x1)
; CHECK-NEXT: Type:     Function
; CHECK-NEXT: Other:    0
; CHECK-NEXT: Section:  .text
; CHECK-NEXT: }

target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @_start() {
  ret void
}

define void @foo() {
  ret void
}
