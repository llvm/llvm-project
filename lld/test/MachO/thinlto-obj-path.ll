; REQUIRES: x86
; RUN: rm -rf %t; split-file %s %t

; RUN: opt -module-summary %t/f.ll -o %t1.o
; RUN: opt -module-summary %t/g.ll -o %t2.o

;; Test to ensure that thinlto-index-only with obj-path creates the file.
; RUN: rm -rf %t4
; RUN: %lld --thinlto-index-only -object_path_lto %t4 -dylib %t1.o %t2.o -o /dev/null
; RUN: llvm-readobj -h %t4/0.x86_64.lto.o | FileCheck %s
; RUN: llvm-nm %t4/0.x86_64.lto.o 2>&1 | FileCheck --check-prefix=NM %s
; RUN: llvm-readobj -h %t4/0.x86_64.lto.o | FileCheck %s

;; Ensure lld emits empty combined module if specific obj-path.
; RUN: rm -fr %t.dir/objpath && mkdir -p %t.dir/objpath
; RUN: %lld -object_path_lto %t4.o -dylib %t1.o %t2.o -o %t.dir/objpath/a.out -save-temps
; RUN: ls %t.dir/objpath/a.out*.lto.* | count 3

;; Ensure lld does not emit empty combined module in default.
; RUN: rm -fr %t.dir/objpath && mkdir -p %t.dir/objpath
; RUN: %lld -dylib %t1.o %t2.o -o %t.dir/objpath/a.out -save-temps
; RUN: ls %t.dir/objpath/a.out*.lto.* | count 2

; NM: no symbols
; CHECK: Format: Mach-O 64-bit x86-64

;--- f.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"

declare void @g(...)

define void @f() {
entry:
  call void (...) @g()
  ret void
}

;--- g.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"

define void @g() {
entry:
  ret void
}
