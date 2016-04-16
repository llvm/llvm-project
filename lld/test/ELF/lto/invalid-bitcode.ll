; REQUIRES: x86
; RUN: llvm-as %s -o %t.o
; RUN: not ld.lld -m elf_x86_64 %t.o 2>&1 | FileCheck %s

; CHECK: invalid bitcode file:

; This bitcode file has no datalayout.
target triple = "x86_64-unknown-linux-gnu"

define void @_start() {
  ret void
}
