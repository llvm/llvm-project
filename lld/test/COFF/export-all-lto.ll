; REQUIRES: x86

; RUN: llvm-as %s -o %t.bc

; RUN: lld-link -lldmingw -dll -out:%t.dll %t.bc -noentry -output-def:%t.def -implib:%t.lib
; RUN: llvm-readobj --coff-exports %t.dll | grep Name: | FileCheck %s
; RUN: cat %t.def | FileCheck --check-prefix=IMPLIB %s
; RUN: llvm-nm %t.lib | FileCheck --check-prefix=IMPLIB-SYMS %s

; CHECK: Name: MyComdatFunc
; CHECK: Name: MyExtData
; CHECK: Name: MyLibFunc

; IMPLIB: MyComdatFunc @1{{$}}
; IMPLIB: MyExtData @2 DATA
; IMPLIB: MyLibFunc @3{{$}}

; IMPLIB-SYMS: 00000000 T MyComdatFunc
; IMPLIB-SYMS: 00000000 T __imp_MyComdatFunc
; IMPLIB-SYMS: 00000000 D __imp_MyExtData
; IMPLIB-SYMS: 00000000 T MyLibFunc
; IMPLIB-SYMS: 00000000 T __imp_MyLibFunc

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-w64-windows-gnu"

@MyExtData = dso_local global i32 42, align 4

$MyComdatFunc = comdat any

define dso_local void @MyLibFunc() {
entry:
  call void @MyComdatFunc()
  ret void
}

define linkonce_odr void @MyComdatFunc() comdat {
entry:
  ret void
}
