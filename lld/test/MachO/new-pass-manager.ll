; REQUIRES: x86

; RUN: llvm-as %s -o %t.o
; RUN: %lld -dylib --lto-debug-pass-manager -o /dev/null %t.o 2>&1 | FileCheck %s

; CHECK: Running pass: GlobalOptPass

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"

define void @foo() {
entry:
  ret void
}
