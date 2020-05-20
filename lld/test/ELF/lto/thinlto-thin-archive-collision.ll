; REQUIRES: x86
; RUN: rm -fr %t && mkdir %t && cd %t
; RUN: mkdir thinlto-archives thinlto-archives/a thinlto-archives/b
; RUN: opt -thinlto-bc -o thinlto-archives/main.o %s
; RUN: opt -thinlto-bc -o thinlto-archives/a/thin.o %S/Inputs/thin1.ll
; RUN: opt -thinlto-bc -o thinlto-archives/b/thin.o %S/Inputs/thin2.ll
; RUN: llvm-ar qcT thinlto-archives/thin.a thinlto-archives/a/thin.o thinlto-archives/b/thin.o
; RUN: ld.lld thinlto-archives/main.o thinlto-archives/thin.a -o thinlto-archives/main.exe --save-temps
; RUN: FileCheck %s < thinlto-archives/main.exe.resolution.txt

; CHECK: thinlto-archives/main.o
; CHECK: thinlto-archives/thin.a(thin.o at {{[1-9][0-9]+}})
; CHECK-NEXT: -r=thinlto-archives/thin.a(thin.o at {{[1-9][0-9]+}}),foo,pl
; CHECK: thinlto-archives/thin.a(thin.o at {{[1-9][0-9]+}})
; CHECK-NEXT: -r=thinlto-archives/thin.a(thin.o at {{[1-9][0-9]+}}),blah,pl

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-scei-ps4"

declare i32 @blah(i32 %meh)
declare i32 @foo(i32 %goo)

define i32 @_start() {
  call i32 @foo(i32 0)
  call i32 @blah(i32 0)
  ret i32 0
}
