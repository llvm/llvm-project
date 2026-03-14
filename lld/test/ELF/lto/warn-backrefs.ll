; REQUIRES: x86
;; Test that the referenced filename is correct (not <internal> or lto.tmp).

; RUN: split-file %s %t
; RUN: llvm-as %t/a.ll -o %ta.o
; RUN: llvm-as %t/b.ll -o %tb.o
; RUN: ld.lld --warn-backrefs --start-lib %tb.o --end-lib %ta.o -o /dev/null 2>&1 | FileCheck %s

; CHECK: warning: backward reference detected: f in {{.*}}a.o refers to {{.*}}b.o

;--- a.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @f()

define void @_start() {
entry:
  call void () @f()
  ret void
}

;--- b.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @f() {
entry:
  ret void
}
