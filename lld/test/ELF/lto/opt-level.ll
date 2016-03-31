; RUN: llvm-as -o %t.o %s
; RUN: ld.lld -o %t0 -m elf_x86_64 -e main --lto-O0 %t.o
; RUN: llvm-nm %t0 | FileCheck --check-prefix=CHECK-O0 %s
; RUN: ld.lld -o %t2 -m elf_x86_64 -e main --lto-O2 %t.o
; RUN: llvm-nm %t2 | FileCheck --check-prefix=CHECK-O2 %s
; RUN: ld.lld -o %t2a -m elf_x86_64 -e main %t.o
; RUN: llvm-nm %t2a | FileCheck --check-prefix=CHECK-O2 %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-O0: foo
; CHECK-O2-NOT: foo
define internal void @foo() {
  ret void
}

define void @main() {
  call void @foo()
  ret void
}
