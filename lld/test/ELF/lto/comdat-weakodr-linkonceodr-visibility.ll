; RUN: rm -rf %t.dir
; RUN: split-file %s %t.dir
; RUN: cd %t.dir
; RUN: echo %t.dir
; RUN: llvm-as explicit.ll -o explicit.bc
; RUN: llvm-as implicit.ll -o implicit.bc


;; Case 1:
; RUN: ld.lld explicit.bc implicit.bc -o case1.so -shared -save-temps
; RUN: llvm-nm case1.so.0.2.internalize.bc | FileCheck %s

;; Case 2:
; RUN: ld.lld implicit.bc explicit.bc -o case2.so -shared -save-temps
; RUN: llvm-nm case2.so.0.2.internalize.bc | FileCheck %s

; CHECK: W foo

;--- explicit.ll
target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

$foo = comdat any
define weak_odr void @foo() local_unnamed_addr comdat {
  ret void
}


;--- implicit.ll
target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

$foo = comdat any
define linkonce_odr void @foo() local_unnamed_addr comdat {
  ret void
}
