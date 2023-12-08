; REQUIRES: x86
; LTO
; RUN: llvm-as %s -o %t.o
; RUN: ld.lld %t.o -o %t.out -wrap=bar -save-temps
; RUN: llvm-readobj --symbols %t.out | FileCheck %s
; RUN: cat %t.out.resolution.txt | FileCheck -check-prefix=RESOLS %s

; ThinLTO
; RUN: opt -module-summary %s -o %t.o
; RUN: ld.lld %t.o -o %t.out -wrap=bar -save-temps
; RUN: llvm-readobj --symbols %t.out | FileCheck %s
; RUN: cat %t.out.resolution.txt | FileCheck -check-prefix=RESOLS %s

; CHECK:      Name: __wrap_bar
; CHECK-NEXT: Value:
; CHECK-NEXT: Size:
; CHECK-NEXT: Binding: Global
; CHECK-NEXT: Type: Function

; Make sure that the 'r' (linker redefined) bit is set for bar and __real_bar
; in the resolutions file. The calls to bar and __real_bar will be routed to
; __wrap_bar and bar, respectively. So they cannot be inlined.
; RESOLS: ,bar,xr{{$}}
; RESOLS: ,__wrap_bar,plx{{$}}
; RESOLS: ,__real_bar,plr{{$}}

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @bar()

define void @_start() {
  call void @bar()
  ret void
}

define void @__wrap_bar() {
  ret void
}

define void @__real_bar() {
  ret void
}
