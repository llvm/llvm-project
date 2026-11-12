; REQUIRES: x86
; RUN: rm -rf %t; split-file %s %t

;; I'm not aware of a deterministic way to verify whether LTO is running in
;; single- or multi-threaded mode. So this test simply checks that we can parse
;; the --thinlto-jobs flag correctly, but doesn't verify its effect.

; RUN: opt -module-summary %t/f.s -o %t/f.o
; RUN: opt -module-summary %t/g.s -o %t/g.o

; RUN: %lld --time-trace --thinlto-jobs=1 -dylib %t/f.o %t/g.o -o %t/out
; RUN: %lld --time-trace --thinlto-jobs=2 -dylib %t/f.o %t/g.o -o %t/out
; RUN: %lld --thinlto-jobs=all -dylib %t/f.o %t/g.o -o /dev/null

;; Test with a bad value
; RUN: not %lld --thinlto-jobs=foo -dylib %t/f.o %t/g.o -o /dev/null 2>&1 | FileCheck %s
; CHECK: error: --thinlto-jobs: invalid job count: foo

;--- f.s
target triple = "x86_64-apple-darwin"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

declare void @g(...)

define void @f() {
entry:
  call void (...) @g()
  ret void
}

;--- g.s
target triple = "x86_64-apple-darwin"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @g() {
entry:
  ret void
}
