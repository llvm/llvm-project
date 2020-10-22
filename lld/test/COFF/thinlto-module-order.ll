; REQUIRES: x86

; RUN: opt -thinlto-bc %s -o %t1.obj
; RUN: opt -thinlto-bc %p/Inputs/thinlto.ll -o %t2.obj

; Ensure module re-ordering in LTO::runThinLTO does not affect the processing order.

; RUN: lld-link -thinlto-index-only:%t3 /entry:main %t1.obj %t2.obj
; RUN: cat %t3 | FileCheck %s --check-prefix=NORMAL
; NORMAL: thinlto-module-order.ll.tmp1.o
; NORMAL: thinlto-module-order.ll.tmp2.o

; RUN: lld-link -thinlto-index-only:%t3 /entry:main %t2.obj %t1.obj 
; RUN: cat %t3 | FileCheck %s --check-prefix=REVERSED
; REVERSED: thinlto-module-order.ll.tmp2.o
; REVERSED: thinlto-module-order.ll.tmp1.o

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24215"

declare void @g(...)

define void @main() {
  call void (...) @g()
  ret void
}
