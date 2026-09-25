; REQUIRES: x86
; RUN: llvm-as %s -o %t.o
; RUN: rm -f %t.missing
; RUN: not ld.lld %t.o -o %t --lto-basic-block-sections=%t.missing 2>&1 | FileCheck %s
; RUN: opt -module-summary %s -o %t.thin.o
; RUN: not ld.lld %t.thin.o -o %t --lto-basic-block-sections=%t.missing 2>&1 | FileCheck %s

; CHECK: error: cannot open {{.*}}.missing:
; CHECK-NOT: PLEASE submit

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
define void @_start() {
  ret void
}
