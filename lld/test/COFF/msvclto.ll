; RUN: llvm-as -o %t1.obj %s
; RUN: llvm-mc -triple=x86_64-pc-windows-msvc -filetype=obj -o %t2.obj %p/Inputs/msvclto.s
; RUN: lld-link %t1.obj %t2.obj /msvclto /out:%t.exe /opt:lldlto=1 /opt:icf \
; RUN:   /entry:main /verbose > %t.log || true
; RUN: FileCheck %s < %t.log

; CHECK: link.exe /nologo {{.*}} {{.*}}2.obj /out:{{.*}}.exe /opt:icf /entry:main /verbose

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

declare void @foo()

define i32 @main() {
  call void @foo()
  ret i32 0
}
