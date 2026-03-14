; REQUIRES: x86
; RUN: rm -rf %t; split-file %s %t
; RUN: llvm-as -o %t/first.bc %t/first.ll
; RUN: llvm-as -o %t/second.bc %t/second.ll
; RUN: %no-fatal-warnings-lld -dylib %t/first.bc %t/second.bc -o /dev/null 2>&1 | FileCheck %s

;; FIXME: can we replace ld-temp.o with a proper name?
; CHECK: warning: linking module flags 'foo': IDs have conflicting values ('i32 2' from {{.*}}second.bc with 'i32 1' from ld-temp.o)

;--- first.ll
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

declare void @f()

define void @g() {
  call void @f()
  ret void
}

!0 = !{ i32 2, !"foo", i32 1 }

!llvm.module.flags = !{ !0 }

;--- second.ll
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @f() {
  ret void
}

!0 = !{ i32 2, !"foo", i32 2 }

!llvm.module.flags = !{ !0 }
