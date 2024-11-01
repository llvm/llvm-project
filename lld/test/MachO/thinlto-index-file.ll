; REQUIRES: x86
; RUN: rm -rf %t; split-file %s %t

; RUN: opt -module-summary %t/f.ll -o %t/1.o
; RUN: opt -module-summary %t/g.ll -o %t/2.o
; RUN: opt -module-summary %t/empty.ll -o %t/3.o

;; Ensure lld writes linked files to linked objects file
; RUN: %lld --thinlto-index-only=%t/1.txt -dylib %t/1.o %t/2.o %t/3.o -o /dev/null
; RUN: FileCheck %s < %t/1.txt
; CHECK: 1.o
; CHECK: 2.o
; CHECK: 3.o

;--- f.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"

declare void @g(...)

define void @f() {
entry:
  call void (...) @g()
  ret void
}

;--- g.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"

define void @g() {
entry:
  ret void
}

;--- empty.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"
