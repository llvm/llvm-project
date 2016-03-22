; REQUIRES: x86
; RUN: llvm-as %p/Inputs/linkonce-odr.ll -o %t1.o
; RUN: llc %s -o %t2.o -filetype=obj
; RUN: ld.lld %t1.o %t2.o -o %t.so -shared
; RUN: llvm-readobj -t %t.so | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
declare void @f()

define void @g() {
  call void @f() ret void
} 

; Be sure that the linkonce_odr symbol 'f' is kept.
; CHECK: Symbol {
; CHECK:   Name: f
; CHECK:   Value: 0x1010
; CHECK:   Size: 1
; CHECK:   Binding: Weak
; CHECK:   Type: Function
; CHECK:   Other: 0
; CHECK:   Section: .text
; CHECK: }
