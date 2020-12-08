; REQUIRES: x86
; RUN: split-file %s %t
; RUN: llvm-as %t/foo.ll -o %t/foo.o
; RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
; RUN: rm -f %t/foo.a
; RUN: llvm-ar rcs %t/foo.a %t/foo.o
; RUN: %lld -save-temps -lSystem %t/test.o %t/foo.a -o %t/test
; RUN: llvm-objdump -d --macho --no-show-raw-insn %t/test | FileCheck %s

; CHECK:      _main:
; CHECK-NEXT: callq _foo
; CHECK-NEXT: retq

;--- foo.ll

target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @foo() {
  ret void
}

;--- test.s

.globl _main
_main:
  callq _foo
  ret
