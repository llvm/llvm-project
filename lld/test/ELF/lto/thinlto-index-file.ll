; REQUIRES: x86
; RUN: rm -rf %t && mkdir %t && cd %t
; RUN: opt -module-summary %s -o 1.o
; RUN: opt -module-summary %p/Inputs/thinlto.ll -o 2.o
; RUN: opt -module-summary %p/Inputs/thinlto_empty.ll -o 3.o

;; Ensure lld writes linked files to linked objects file
; RUN: ld.lld --plugin-opt=thinlto-index-only=1.txt -shared 1.o 2.o 3.o -o /dev/null
; RUN: FileCheck %s < 1.txt
; CHECK: 1.o
; CHECK: 2.o
; CHECK: 3.o

;; Check that this also works without the --plugin-opt= prefix.
; RUN: ld.lld --thinlto-index-only=2.txt -shared 1.o 2.o 3.o -o /dev/null
; RUN: diff 1.txt 2.txt

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @g(...)

define void @f() {
entry:
  call void (...) @g()
  ret void
}
