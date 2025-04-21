; REQUIRES: x86
; RUN: rm -rf %t; split-file %s %t

; RUN: llvm-as %t/lib.ll -o %t/lib.o
; RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/main.o %t/main.s
; RUN: llvm-ar rcST %t/lib.a %t/lib.o
; RUN: %lld %t/main.o %t/lib.a -o %t/out

;--- main.s
.global _main
_main:
    call _foo    
    mov $0, %rax
    ret

;--- lib.ll
target triple = "x86_64-apple-darwin"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @foo() {
entry:
  ret void
}
