; REQUIRES: x86
; RUN: llvm-as %s -o %t.o
; RUN: ld.lld -m elf_x86_64 %t.o -o %t2 --export-dynamic -save-temps
; RUN: llvm-dis < %t2.lto.bc | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @_start() {
  ret void
}

define hidden void @foo() {
  ret void
}

; Check that _start and foo are not internalized.
; CHECK: define void @_start()
; CHECK: define hidden void @foo()
