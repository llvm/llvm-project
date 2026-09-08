; REQUIRES: x86
;; Herbception ("throws"/"fails{E}") ODR checking during ThinLTO: the check
;; runs on the backend threads, which parse every module.

; RUN: rm -rf %t.dir && mkdir %t.dir
; RUN: opt -module-summary %p/Inputs/herbception-odr-throws.ll -o %t.dir/throws.o
; RUN: opt -module-summary %p/Inputs/herbception-odr-plain-decl.ll -o %t.dir/decl.o

; RUN: not ld.lld --thinlto-jobs=2 %t.dir/throws.o %t.dir/decl.o -o /dev/null 2>&1 | FileCheck %s

; CHECK: error: herbception ODR violation: symbol 'f' has conflicting definitions: it is defined as a herbception ('throws') function with error payload type '{ ptr, i64 }' in '{{.*}}throws.o', but without the herbception error channel in '{{.*}}decl.o'

;; Consistent signatures link fine.
; RUN: opt -module-summary %s -o %t.dir/start.o
; RUN: ld.lld %t.dir/throws.o %t.dir/start.o -o %tok

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @_start() {
entry:
  %r = call { { ptr, i64 }, i1 } @f()
  ret void
}

declare { { ptr, i64 }, i1 } @f() #0

attributes #0 = { throws }
