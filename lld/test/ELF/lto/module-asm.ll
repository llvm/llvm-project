; REQUIRES: x86
; RUN: llvm-as %s -o %t.o
; RUN: ld.lld %t.o -o %t
; RUN: llvm-nm %t | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

module asm ".text"
module asm ".globl foo"
; CHECK: T foo
module asm "foo: ret"

declare void @foo()

define void @_start() {
  call void @foo()
  ret void
}
