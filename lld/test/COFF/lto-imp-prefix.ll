; REQUIRES: x86

; RUN: rm -rf %t.dir
; RUN: split-file %s %t.dir
; RUN: llvm-as %t.dir/main.ll -o %t.main.obj
; RUN: llvm-as %t.dir/other1.ll -o %t.other1.obj
; RUN: llvm-as %t.dir/other2.ll -o %t.other2.obj

; RUN: lld-link /entry:entry %t.main.obj %t.other1.obj /out:%t1.exe /subsystem:console /debug:symtab

;; Check that we don't retain __imp_ prefixed symbols we don't need.
; RUN: llvm-nm %t1.exe | FileCheck %s
; CHECK-NOT: __imp_unusedFunc

; RUN: lld-link /entry:entry %t.main.obj %t.other2.obj /out:%t2.exe /subsystem:console

;--- main.ll
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-w64-windows-gnu"

define void @entry() {
entry:
  tail call void @importedFunc()
  tail call void @other()
  ret void
}

declare dllimport void @importedFunc()

declare void @other()

;--- other1.ll
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-w64-windows-gnu"

@__imp_importedFunc = global ptr @importedFuncReplacement

define internal void @importedFuncReplacement() {
entry:
  ret void
}

@__imp_unusedFunc = global ptr @unusedFuncReplacement

define internal void @unusedFuncReplacement() {
entry:
  ret void
}

define void @other() {
entry:
  ret void
}

;--- other2.ll
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-w64-windows-gnu"

@__imp_importedFunc = global ptr @importedFunc

; Test with two external symbols with the same name, with/without the __imp_
; prefix.
define void @importedFunc() {
entry:
  ret void
}

define void @other() {
entry:
  ret void
}
