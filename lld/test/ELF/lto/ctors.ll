; REQUIRES: x86
; RUN: llvm-as %s -o %t.o
; RUN: ld.lld -m elf_x86_64 %t.o -o %t.so -shared
; RUN: llvm-readobj -sections %t.so | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @ctor, i8* null }]
define void @ctor() {
  ret void
}

; The llvm.global_ctors should end up producing constructors.
; CHECK: Name: .ctors
