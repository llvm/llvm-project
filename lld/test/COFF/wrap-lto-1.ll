; REQUIRES: x86
; LTO
; RUN: llvm-as %s -o %t.obj
; RUN: lld-link -out:%t.exe %t.obj -entry:entry -subsystem:console -wrap:bar -debug:symtab -lldsavetemps
; RUN: cat %t.exe.resolution.txt | FileCheck -check-prefix=RESOLS %s

; ThinLTO
; RUN: opt -module-summary %s -o %t.obj
; RUN: lld-link -out:%t.exe %t.obj -entry:entry -subsystem:console -wrap:bar -debug:symtab -lldsavetemps
; RUN: cat %t.exe.resolution.txt | FileCheck -check-prefix=RESOLS %s

; Make sure that the 'r' (linker redefined) bit is set for bar and __real_bar
; in the resolutions file. The calls to bar and __real_bar will be routed to
; __wrap_bar and bar, respectively. So they cannot be inlined.
; RESOLS: ,bar,pxr{{$}}
; RESOLS: ,__real_bar,xr{{$}}
; RESOLS: ,__wrap_bar,px{{$}}

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-w64-windows-gnu"

define void @bar() {
  ret void
}

define void @entry() {
  call void @bar()
  ret void
}

declare void @__real_bar()

define void @__wrap_bar() {
  call void @__real_bar()
  ret void
}
