; REQUIRES: x86

;; Verify that we run the ObjCARCContractPass during LTO. Without that, the
;; objc.clang.arc.use intrinsic will get passed to the instruction selector,
;; which doesn't know how to handle it.

; RUN: llvm-as %s -o %t.o
; RUN: %lld -dylib -lSystem %t.o -o %t --no-lto-legacy-pass-manager
; RUN: llvm-objdump -d %t | FileCheck %s

; RUN: opt -module-summary %s -o %t.o
; RUN: %lld -dylib -lSystem %t.o -o %t --no-lto-legacy-pass-manager
; RUN: llvm-objdump -d %t | FileCheck %s

; CHECK:      <_foo>:
; CHECK-NEXT: retq

target triple = "x86_64-apple-darwin"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(ptr %a, ptr %b) {
  call void (...) @llvm.objc.clang.arc.use(ptr %a, ptr %b) nounwind
  ret void
}

declare void @llvm.objc.clang.arc.use(...) nounwind
