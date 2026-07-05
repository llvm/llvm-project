; REQUIRES: x86
;; Similar to ../wrap-defined.s but for LTO.

; RUN: llvm-as %s -o %t.o
; RUN: ld.lld -shared %t.o -wrap=bar -o %t.so
; RUN: llvm-objdump -d %t.so | FileCheck %s

; CHECK:      <_start>:
; CHECK-NEXT:   jmp {{.*}} <__wrap_bar@plt>

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @bar() {
  ret void
}

define void @_start() {
  call void @bar()
  ret void
}
