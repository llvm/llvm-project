;; Make sure we do not pass archive files containing bitcode files.

; RUN: llvm-as -o %t.obj %s
; RUN: llvm-ar cru %t-main.a %t.obj
; RUN: mkdir -p %t.dir
; RUN: llvm-mc -triple=x86_64-pc-windows-msvc -filetype=obj -o %t.dir/bitcode.obj %p/Inputs/msvclto.s
; RUN: lld-link %t-main.a %t.dir/bitcode.obj /msvclto /out:%t.exe /opt:lldlto=1 /opt:icf \
; RUN:   /entry:main /verbose > %t.log || true
; RUN: FileCheck %s < %t.log

; CHECK-NOT: link.exe {{.*}}t-main.a

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

declare void @foo()

define i32 @main() {
  call void @foo()
  ret i32 0
}
