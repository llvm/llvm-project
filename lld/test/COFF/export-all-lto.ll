; REQUIRES: x86

; RUN: llvm-as %s -o %t.bc

; RUN: lld-link -lldmingw -dll -out:%t.dll %t.bc -noentry -output-def:%t.def
; RUN: llvm-readobj --coff-exports %t.dll | grep Name: | FileCheck %s
; RUN: cat %t.def | FileCheck --check-prefix=IMPLIB %s

; CHECK: Name: MyExtData
; CHECK: Name: MyLibFunc

; IMPLIB: MyExtData @1 DATA
; IMPLIB: MyLibFunc @2{{$}}

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-w64-windows-gnu"

@MyExtData = dso_local global i32 42, align 4

define dso_local void @MyLibFunc() {
entry:
  ret void
}
