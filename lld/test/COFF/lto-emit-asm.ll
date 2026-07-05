; REQUIRES: x86
; RUN: llvm-as %s -o %t.obj

; RUN: lld-link /lldemit:asm /dll /noentry /include:f1 /include:f2 %t.obj /opt:lldltopartitions=1 /out:%t.1p /lldsavetemps
; RUN: cat %t.1p.lto.s | FileCheck %s
; RUN: llvm-dis %t.1p.0.4.opt.bc -o - | FileCheck --check-prefix=OPT %s

; RUN: lld-link /lldemit:asm /dll /noentry /include:f1 /include:f2 %t.obj /opt:lldltopartitions=2 /out:%t.2p
; RUN: cat %t.2p.lto.s %t.2p.lto.1.s | FileCheck %s

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

;; Note: we also check for the presence of comments; /lldemit:asm output should be verbose.

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
