; REQUIRES: x86
; RUN: llvm-as %s -o %t.o
; RUN: ld.lld -m elf_x86_64 %t.o -o %t
; RUN: llvm-objdump -d %t | FileCheck %s

; CHECK: _start:
; CHECK-NEXT: retq

; This bitcode file has no datalayout.
; Check that we produce a valid binary out of it.
target triple = "x86_64-unknown-linux-gnu"

define void @_start() {
  ret void
}
