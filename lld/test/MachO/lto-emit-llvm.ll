; REQUIRES: x86
;
; Check that the --lto-emit-llvm option is handled correctly.
;
; RUN: opt %s -o %t.o
; RUN: ld.lld --lto-emit-llvm %t.o -o %t.out.o
; RUN: llvm-dis < %t.out.o -o - | FileCheck %s
;
; CHECK: define hidden void @main()

target triple = "x86_64-apple-darwin"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

@llvm.compiler.used = appending global [1 x ptr] [ptr @main], section "llvm.metadata"

define hidden void @main() {
  ret void
}
