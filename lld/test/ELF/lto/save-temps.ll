; REQUIRES: x86
; RUN: rm -f %t %t.lto.bc %t.lto.o
; RUN: llvm-as %s -o %t.o
; RUN: llvm-as %p/Inputs/save-temps.ll -o %t2.o
; RUN: ld.lld -shared -m elf_x86_64 %t.o %t2.o -o %t -save-temps
; RUN: llvm-nm %t | FileCheck %s
; RUN: llvm-nm %t.lto.bc | FileCheck %s
; RUN: llvm-nm %t.lto.o | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo() {
  ret void
}

; CHECK: T bar
; CHECK: T foo
