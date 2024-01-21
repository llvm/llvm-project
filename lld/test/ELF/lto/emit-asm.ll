; REQUIRES: x86
; RUN: rm -rf %t && mkdir %t && cd %t
; RUN: llvm-as %s -o a.bc
; RUN: ld.lld --lto-emit-asm -shared a.bc -o - | FileCheck %s
; RUN: ld.lld --plugin-opt=emit-asm --plugin-opt=lto-partitions=2 -shared a.bc -o out.s
; RUN: cat out.s out.s1 | FileCheck %s

; RUN: ld.lld --lto-emit-asm --save-temps -shared a.bc -o out.s
; RUN: FileCheck --input-file out.s %s
; RUN: llvm-dis out.s.0.4.opt.bc -o - | FileCheck --check-prefix=OPT %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

;; Note: we also check for the presence of comments; --lto-emit-asm output should be verbose.

; CHECK-DAG: # -- Begin function f1
; CHECK-DAG: f1:
; OPT: define void @f1()
define void @f1() {
  ret void
}

; CHECK-DAG: # -- Begin function f2
; CHECK-DAG: f2:
; OPT: define void @f2()
define void @f2() {
  ret void
}
