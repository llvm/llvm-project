; REQUIRES: x86
;; LTO-generated relocatable files may reference _GLOBAL_OFFSET_TABLE_ while
;; the IR does not mention _GLOBAL_OFFSET_TABLE_.
;; Test that there is no spurious "undefined symbol" error.

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: opt -module-summary b.ll -o b.bc

;; Test Thin LTO.
; RUN: cat a.ll medium.ll | opt -module-summary - -o medium.bc
; RUN: ld.lld -pie --no-relax medium.bc b.bc -o medium
; RUN: llvm-objdump -dt medium | FileCheck %s

;; Test regular LTO.
; RUN: cat a.ll large.ll | llvm-as - -o large.bc
; RUN: ld.lld -pie large.bc b.bc -o large
; RUN: llvm-objdump -dt large | FileCheck %s

;; Explicit reference of _GLOBAL_OFFSET_TABLE_ is fine.
; RUN: cat a.ll medium.ll ref.ll | opt -module-summary - -o ref.bc
; RUN: ld.lld -pie -u ref ref.bc b.bc -y _GLOBAL_OFFSET_TABLE_ -o ref 2>&1 | FileCheck %s --check-prefix=TRACE
; RUN: llvm-objdump -dt ref | FileCheck %s

; TRACE:      ref.bc: reference to _GLOBAL_OFFSET_TABLE_
; TRACE-NEXT: ref.bc: reference to _GLOBAL_OFFSET_TABLE_
; TRACE-NEXT: <internal>: definition of _GLOBAL_OFFSET_TABLE_
; TRACE-NEXT: ref.lto.ref.o: reference to _GLOBAL_OFFSET_TABLE_

;; The IR symbol table references _GLOBAL_OFFSET_TABLE_, which causes lld to define the symbol.
; CHECK: .got.plt       0000000000000000 .hidden _GLOBAL_OFFSET_TABLE_
; CHECK: movabsq

;--- a.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@i = external global i32

define dso_local void @_start() {
entry:
  %0 = load i32, ptr @i
  %inc = add nsw i32 %0, 1
  store i32 %inc, ptr @i
  ret void
}

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 8, !"PIC Level", i32 2}
!1 = !{i32 7, !"PIE Level", i32 2}
!2 = !{i32 1, !"Large Data Threshold", i64 0}

;--- medium.ll
!3 = !{i32 1, !"Code Model", i32 3}

;--- large.ll
!3 = !{i32 1, !"Code Model", i32 4}

;--- ref.ll
@_GLOBAL_OFFSET_TABLE_ = external global [0 x i8]

define dso_local ptr @ref() {
entry:
  ret ptr @_GLOBAL_OFFSET_TABLE_
}

;--- b.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@i = global i32 0
