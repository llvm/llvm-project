; REQUIRES: x86
;
; RUN: llvm-as %s -o %t1.o
; RUN: llvm-as %p/Inputs/start-lib.ll -o %t2.o
;
; RUN: ld.lld -m elf_x86_64 -shared -o %t3 %t1.o %t2.o
; RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=ADDED %s
; ADDED: Name: _bar
;
; RUN: ld.lld -m elf_x86_64 -shared -o %t3 %t1.o --start-lib %t2.o
; RUN: llvm-readobj --symbols %t3 | FileCheck --check-prefix=LIB %s
; LIB-NOT: Name: _bar

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @_start() {
  ret void
}
