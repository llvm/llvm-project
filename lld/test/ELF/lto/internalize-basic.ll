; REQUIRES: x86
; RUN: llvm-as %s -o %t.o
; RUN: ld.lld %t.o -o %t2 -save-temps
; RUN: llvm-dis < %t2.0.2.internalize.bc | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @_start() {
  call void @bar()
  ret void
}

define void @bar() {
  ret void
}

define hidden void @foo() {
  ret void
}

; Check that _start is not internalized.
; CHECK: define dso_local void @_start()

; Check that the foo and bar functions are correctly internalized.
; CHECK: define internal void @bar()
; CHECK: define internal void @foo()
