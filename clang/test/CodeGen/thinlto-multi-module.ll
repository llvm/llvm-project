; REQUIRES: x86-registered-target

; RUN: opt -module-summary -o %t1.o %s
; RUN: llvm-lto -thinlto -o %t %t1.o

; RUN: opt -module-summary -o %t2.o %S/Inputs/thinlto-multi-module.ll
; RUN: llvm-cat -b -o %t1cat.o %t2.o %t1.o
; RUN: cp %t1cat.o %t1.o
; RUN: %clang -target x86_64-unknown-linux-gnu -O2 -o %t3.o -x ir %t1.o -c -fthinlto-index=%t.thinlto.bc
; RUN: llvm-nm %t3.o | FileCheck --check-prefix=CHECK-OBJ %s
; CHECK-OBJ: T f1
; CHECK-OBJ: U f2

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @f2()

define void @f1() {
  call void @f2()
  ret void
}
