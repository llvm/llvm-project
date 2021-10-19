; REQUIRES: x86
; RUN: llvm-as %s -o %t.o

; RUN: ld.lld %t.o -o %t -save-temps --export-dynamic --noinhibit-exec
; RUN: llvm-readobj -r %t.lto.o | FileCheck %s --check-prefix=STATIC

; RUN: ld.lld %t.o -o %t -save-temps -r -mllvm -relocation-model=static
; RUN: llvm-readobj -r %t.lto.o | FileCheck %s --check-prefix=STATIC

; STATIC: R_X86_64_PC32 foo

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@foo = external dso_local global i32
define i32 @main() {
  %t = load i32, i32* @foo
  ret i32 %t
}
