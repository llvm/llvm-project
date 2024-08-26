; REQUIRES: x86
;; malloc+memset references can be combined to a calloc reference.
;; Test that we extract calloc defined in a lazy bitcode file to satisfy
;; possible references from LTO generated object files.

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: llvm-as a.ll -o a.bc
; RUN: llvm-as calloc.ll -o calloc.bc
; RUN: llvm-mc -filetype=obj -triple=x86_64 lib.s -o lib.o
; RUN: ld.lld -u foo a.bc --start-lib calloc.bc lib.o --end-lib -o t --save-temps
; RUN: llvm-dis < t.0.4.opt.bc | FileCheck %s
; RUN: llvm-nm t | FileCheck %s --check-prefix=NM

; CHECK: define dso_local void @calloc

; NM-NOT:  {{.}}
; NM:      {{.*}} T _start
;; TODO: Currently the symbol is lazy, which lowers to a SHN_ABS symbol at address 0.
; NM-NEXT: {{.*}} T calloc
; NM-NEXT: {{.*}} T foo
; NM-NEXT: {{.*}} T malloc
; NM-NEXT: {{.*}} T memset
; NM-NOT:  {{.}}

;--- a.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define ptr @foo() noinline {
entry:
  %call = tail call noalias ptr @malloc(i64 400)
  tail call void @llvm.memset.p0.i64(ptr %call, i8 0, i64 400, i1 false)
  ret ptr %call
}

define void @_start(ptr %a, ptr %b) {
entry:
  call ptr @foo()
  ret void
}

declare ptr @malloc(i64)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)

;--- calloc.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @calloc(i64, i64) {
entry:
  ret void
}

;--- lib.s
.globl malloc, memset
malloc:
memset:
