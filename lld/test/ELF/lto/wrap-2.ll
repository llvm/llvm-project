; REQUIRES: x86
;; This test verifies that --wrap works correctly for inter-module references to
;; the wrapped symbol, when LTO or ThinLTO is involved. It checks for various
;; combinations of bitcode and regular objects.

;; LTO + LTO
; RUN: llvm-as %s -o %t1.bc
; RUN: llvm-as %S/Inputs/wrap-bar.ll -o %t2.bc
; RUN: ld.lld %t1.bc %t2.bc -shared -o %t.bc-bc.so -wrap=bar
; RUN: llvm-objdump -d %t.bc-bc.so | FileCheck %s --check-prefixes=CHECK,JMP
; RUN: llvm-readobj --symbols %t.bc-bc.so | FileCheck --check-prefix=BIND %s

;; LTO + Object
; RUN: llc %S/Inputs/wrap-bar.ll -o %t2.o --filetype=obj
; RUN: ld.lld %t1.bc %t2.o -shared -o %t.bc-o.so -wrap=bar
; RUN: llvm-objdump -d %t.bc-o.so | FileCheck %s --check-prefixes=CHECK,JMP
; RUN: llvm-readobj --symbols %t.bc-o.so | FileCheck --check-prefix=BIND %s

;; Object + LTO
; RUN: llc %s -o %t1.o --filetype=obj
; RUN: ld.lld %t1.o %t2.bc -shared -o %t.o-bc.so -wrap=bar
; RUN: llvm-objdump -d %t.o-bc.so | FileCheck %s --check-prefixes=CHECK,CALL
; RUN: llvm-readobj --symbols %t.o-bc.so | FileCheck --check-prefix=BIND %s

;; ThinLTO + ThinLTO
; RUN: opt -module-summary %s -o %t1.thin
; RUN: opt -module-summary %S/Inputs/wrap-bar.ll -o %t2.thin
; RUN: ld.lld %t1.thin %t2.thin -shared -o %t.thin-thin.so -wrap=bar
; RUN: llvm-objdump -d %t.thin-thin.so | FileCheck %s --check-prefixes=CHECK,JMP
; RUN: llvm-readobj --symbols %t.thin-thin.so | FileCheck --check-prefix=BIND %s

;; ThinLTO + Object
; RUN: ld.lld %t1.thin %t2.o -shared -o %t.thin-o.so -wrap=bar
; RUN: llvm-objdump -d %t.thin-o.so | FileCheck %s --check-prefixes=CHECK,JMP
; RUN: llvm-readobj --symbols %t.thin-o.so | FileCheck --check-prefix=BIND %s

;; Object + ThinLTO
; RUN: ld.lld %t1.o %t2.thin -shared -o %t.o-thin.so -wrap=bar
; RUN: llvm-objdump -d %t.o-thin.so | FileCheck %s --check-prefixes=CHECK,CALL
; RUN: llvm-readobj --symbols %t.o-thin.so | FileCheck --check-prefix=BIND %s

;; Make sure that calls in foo() are not eliminated and that bar is
;; routed to __wrap_bar and __real_bar is routed to bar.

; CHECK:      <foo>:
; CHECK-NEXT: pushq	%rax
; CHECK-NEXT: callq{{.*}}<__wrap_bar>
; JMP-NEXT:   popq  %rax
; JMP-NEXT:   jmp{{.*}}<bar>
; CALL-NEXT:  callq{{.*}}<bar>
; CALL-NEXT:  popq  %rax

;; Check that bar and __wrap_bar retain their original binding.
; BIND:      Name: bar
; BIND-NEXT: Value:
; BIND-NEXT: Size:
; BIND-NEXT: Binding: Local
; BIND:      Name: __wrap_bar
; BIND-NEXT: Value:
; BIND-NEXT: Size:
; BIND-NEXT: Binding: Local

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @bar()
declare void @__real_bar()

define void @foo() {
  call void @bar()
  call void @__real_bar()
  ret void
}
