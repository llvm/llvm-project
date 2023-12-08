; REQUIRES: x86

; RUN: opt -module-hash -module-summary %s -o %t.o
; RUN: ld.lld --plugin-opt=emit-llvm -o %t.out.o %t.o
; RUN: llvm-dis < %t.out.o -o - | FileCheck %s

;; Regression test for D112297: bitcode writer used to crash when
;; --plugin-opt=emit-llvmis enabled and the output is /dev/null.
; RUN: ld.lld --plugin-opt=emit-llvm -mllvm -bitcode-flush-threshold=0 -o /dev/null %t.o

; CHECK: define internal void @main()

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @main() {
  ret void
}
