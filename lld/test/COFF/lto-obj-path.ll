; REQUIRES: x86

; Test to ensure that thinlto-index-only with lto-obj-path creates
; the native object file.
; RUN: opt -module-summary %s -o %t1.obj
; RUN: opt -module-summary %p/Inputs/thinlto.ll -o %t2.obj
; RUN: rm -f %t4.obj
; RUN: lld-link -thinlto-index-only -lto-obj-path:%t4.obj -out:t3.exe \
; RUN:     -entry:main %t1.obj %t2.obj
; RUN: llvm-readobj -h %t4.obj | FileCheck %s
; RUN: llvm-nm %t4.obj 2>&1 | FileCheck %s -check-prefix=SYMBOLS
; RUN: llvm-nm %t4.obj 2>&1 | count 1

; CHECK: Format: COFF-x86-64
; SYMBOLS: @feat.00

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24215"

declare void @g(...)

define void @main() {
  call void (...) @g()
  ret void
}
