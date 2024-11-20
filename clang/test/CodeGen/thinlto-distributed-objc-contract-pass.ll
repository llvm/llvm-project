; RUN: opt -thinlto-bc -o %t.o %s

; RUN: llvm-lto2 run -thinlto-distributed-indexes %t.o \
; RUN:   -o %t2.index \
; RUN:   -r=%t.o,_use_arc,px

; RUN: %clang_cc1 -O2 -triple x86_64-apple-darwin \
; RUN:   -emit-obj -fthinlto-index=%t.o.thinlto.bc \
; RUN:   -o %t.native.o -x ir %t.o

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"

define void @use_arc(ptr %a, ptr %b) {
  call void (...) @llvm.objc.clang.arc.use(ptr %a, ptr %b) nounwind
  ret void
}

declare void @llvm.objc.clang.arc.use(...) nounwind
