; REQUIRES: x86
; RUN: llvm-as %s -o %t.o
; RUN: %lld %t.o -o %t
; RUN: llvm-objdump -d %t | FileCheck %s

; CHECK:      <_foo>:
; CHECK-NEXT: retq

; CHECK:      <_main>:
; CHECK-NEXT: jmp {{.*}} <_foo>

target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

module asm ".text"
module asm ".globl _foo"
module asm "_foo: ret"

declare void @foo()

define void @main() {
  call void @foo()
  ret void
}
