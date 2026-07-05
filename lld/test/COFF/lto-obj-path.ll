; REQUIRES: x86

; Test to ensure that thinlto-index-only with lto-obj-path creates
; the native object file.
; RUN: rm -rf %t.dir/objpath && mkdir -p %t.dir/objpath
; RUN: opt -module-summary %s -o %t.dir/objpath/t1.obj
; RUN: opt -module-summary %p/Inputs/thinlto.ll -o %t.dir/objpath/t2.obj
; RUN: rm -f %t.dir/objpath/t4.obj
; RUN: lld-link -thinlto-index-only -lto-obj-path:%t.dir/objpath/t4.obj \
; RUN:     -out:%t.dir/objpath/t3.exe -entry:main %t.dir/objpath/t1.obj \
; RUN:     %t.dir/objpath/t2.obj
; RUN: llvm-readobj -h %t.dir/objpath/t4.obj | FileCheck %s
; RUN: llvm-nm %t.dir/objpath/t4.obj 2>&1 | FileCheck %s -check-prefix=SYMBOLS
; RUN: llvm-nm %t.dir/objpath/t4.obj 2>&1 | count 1

;; Ensure lld emits empty combined module if specific obj-path.
; RUN: lld-link /out:%t.dir/objpath/a.exe -lto-obj-path:%t.dir/objpath/t4.obj \
; RUN:     -entry:main %t.dir/objpath/t1.obj %t.dir/objpath/t2.obj -lldsavetemps
; RUN: ls %t.dir/objpath/a.exe.lto.obj
; RUN: ls %t.dir/objpath/a.exe.lto.t1.obj
; RUN: ls %t.dir/objpath/a.exe.lto.t2.obj

;; Ensure lld does not emit empty combined module in default.
; RUN: rm %t.dir/objpath/a.exe.lto.*
; RUN: lld-link /out:%t.dir/objpath/a.exe \
; RUN:     -entry:main %t.dir/objpath/t1.obj %t.dir/objpath/t2.obj -lldsavetemps
; RUN: ls %t.dir/objpath/a.exe.lto.t1.obj
; RUN: ls %t.dir/objpath/a.exe.lto.t2.obj
; RUN: not ls %t.dir/objpath/a.exe.lto.obj

; CHECK: Format: COFF-x86-64
; SYMBOLS: @feat.00

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24215"

declare void @g(...)

define void @main() {
  call void (...) @g()
  ret void
}
