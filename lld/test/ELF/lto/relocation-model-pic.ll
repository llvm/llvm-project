; REQUIRES: x86
; RUN: llvm-as %s -o %t.o

; RUN: ld.lld %t.o -o %t -save-temps -shared
; RUN: llvm-readobj -r %t.lto.o | FileCheck %s

; RUN: ld.lld %t.o -o %t -save-temps --export-dynamic -pie -z undefs
; RUN: llvm-readobj -r %t.lto.o | FileCheck %s

; RUN: ld.lld %t.o -o %t -save-temps --export-dynamic -z undefs
; RUN: llvm-readobj -r %t.lto.o | FileCheck %s

; CHECK: R_X86_64_REX_GOTPCRELX foo

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@foo = external global i32
define i32 @main() {
  %t = load i32, ptr @foo
  ret i32 %t
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"PIC Level", i32 2}
