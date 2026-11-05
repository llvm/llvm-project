; REQUIRES: x86
; RUN: rm -rf %t && split-file %s %t

; The X86 backend adds a reference to _fltused when a module uses floating
; point, which is not in the bitcode symbol table. An archive member that
; defines it as bitcode has to be added to the link before LTO runs.

; RUN: llvm-as -o %t/main64.obj %t/main64.ll
; RUN: llvm-as -o %t/lib64.obj %t/lib64.ll
; RUN: llvm-ar rcs %t/lib64.a %t/lib64.obj
; RUN: lld-link -out:%t/out64.exe -subsystem:console -entry:main -opt:noref -lldmap:- %t/main64.obj %t/lib64.a | FileCheck %s --check-prefix=X64

; RUN: llvm-as -o %t/main32.obj %t/main32.ll
; RUN: llvm-as -o %t/lib32.obj %t/lib32.ll
; RUN: llvm-ar rcs %t/lib32.a %t/lib32.obj
; RUN: lld-link -out:%t/out32.exe -subsystem:console -entry:main -safeseh:no -opt:noref -lldmap:- %t/main32.obj %t/lib32.a | FileCheck %s --check-prefix=X86

; X64: _fltused
; X86: __fltused

;--- main64.ll
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define double @main(double %a) {
  %r = fadd double %a, 1.0
  ret double %r
}

;--- lib64.ll
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

@_fltused = global i32 0

;--- main32.ll
target datalayout = "e-m:x-p:32:32-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc"

define double @main(double %a) {
  %r = fadd double %a, 1.0
  ret double %r
}

;--- lib32.ll
target datalayout = "e-m:x-p:32:32-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc"

@_fltused = global i32 0
