; REQUIRES: x86-registered-target
; Test that mixing bitcode file with opaque and typed pointers works.

; RUN: mkdir -p %t
; RUN: opt -module-summary -o %t/typed.bc %s
; RUN: opt -module-summary -o %t/opaque.bc %S/Inputs/thinlto-opaque.ll
; RUN: llvm-lto2 run -thinlto-distributed-indexes %t/typed.bc %t/opaque.bc \
; RUN:   -o %t/native.o -r %t/typed.bc,main,plx -r %t/typed.bc,f2, \
; RUN:   -r %t/opaque.bc,f2,p

; RUN: %clang_cc1 -triple x86_64-- -emit-obj -o %t/native.o %t/typed.bc \
; RUN:   -Wno-override-module \
; RUN:   -fthinlto-index=%t/typed.bc.thinlto.bc

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64--"

declare ptr @f2()

define i32 @main() {
  call ptr @f2()
  ret i32 0
}
