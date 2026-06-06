; REQUIRES: x86
; RUN: rm -rf %t && mkdir %t
; RUN: llvm-as -o %t/lto.obj %s

; RUN: lld-link /lldemit:llvm /out:%t/lto.bc /entry:main /subsystem:console %t/lto.obj
; RUN: llvm-dis %t/lto.bc -o - | FileCheck %s

; CHECK: define void @main() local_unnamed_addr

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define void @main() {
  ret void
}
