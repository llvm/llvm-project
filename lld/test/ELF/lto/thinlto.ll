; Basic ThinLTO tests.
; RUN: llvm-as %s -o %t.o
; RUN: llvm-as %p/Inputs/thinlto.ll -o %t2.o

; First force single-threaded mode
; RUN: ld.lld -save-temps --lto-jobs=1 -shared %t.o %t2.o -o %t
; RUN: llvm-nm %t.lto.o | FileCheck %s --check-prefix=NM

; NM: T f
; NM: T g

; Next force multi-threaded mode
; RUN: ld.lld -save-temps --lto-jobs=2 -shared %t.o %t2.o -o %t2
; RUN: llvm-nm %t20.lto.o | FileCheck %s --check-prefix=NM1
; RUN: llvm-nm %t21.lto.o | FileCheck %s --check-prefix=NM2

; NM1: T g
; NM2: T f

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @g(...)

define void @f() {
entry:
  call void (...) @g()
  ret void
}
