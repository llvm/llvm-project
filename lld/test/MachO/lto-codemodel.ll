; REQUIRES: x86
; RUN: llvm-as %s -o %t.o
; RUN: %lld -lSystem %t.o -o %ts -mllvm -code-model=small
; RUN: %lld -lSystem %t.o -o %tl -mllvm -code-model=large
; RUN: llvm-objdump --no-print-imm-hex -d %ts | FileCheck %s --check-prefix=CHECK-SMALL
; RUN: llvm-objdump --no-print-imm-hex -d %tl | FileCheck %s --check-prefix=CHECK-LARGE

target triple = "x86_64-apple-darwin"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

@data = internal constant [0 x i32] []

define i32* @main() nounwind readonly {
entry:
; CHECK-SMALL-LABEL: <_main>:
; CHECK-SMALL:       leaq    [[#]](%rip), %rax
; CHECK-LARGE-LABEL: <_main>:
; CHECK-LARGE:       movabsq $[[#]], %rax
  ret i32* getelementptr ([0 x i32], [0 x i32]* @data, i64 0, i64 0)
}
