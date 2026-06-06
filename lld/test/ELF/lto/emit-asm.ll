; REQUIRES: x86
; RUN: rm -rf %t && mkdir %t && cd %t
; RUN: llvm-as %s -o a.bc
; RUN: ld.lld --lto-emit-asm -shared a.bc -o out 2>&1 | count 0
; RUN: FileCheck %s < out.lto.s
; RUN: ld.lld --plugin-opt=emit-asm --plugin-opt=lto-partitions=2 -shared a.bc -o out
; RUN: cat out.lto.s out.lto.1.s | FileCheck %s

; RUN: ld.lld --lto-emit-asm --save-temps -shared a.bc -o out
; RUN: FileCheck --input-file out.lto.s %s
; RUN: llvm-dis out.0.4.opt.bc -o - | FileCheck --check-prefix=OPT %s

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
