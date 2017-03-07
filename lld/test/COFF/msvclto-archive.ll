;; Make sure we do not pass archive files containing only bitcode files.

; RUN: llvm-as -o %t.obj %s
; RUN: llvm-ar cru %t-main1.a %t.obj
; RUN: mkdir -p %t.dir
; RUN: llvm-mc -triple=x86_64-pc-windows-msvc -filetype=obj -o %t.dir/bitcode.obj %p/Inputs/msvclto.s
; RUN: lld-link %t-main1.a %t.dir/bitcode.obj /msvclto /out:%t.exe /opt:lldlto=1 /opt:icf \
; RUN:   /entry:main /verbose > %t.log || true
; RUN: FileCheck -check-prefix=BC %s < %t.log
; BC-NOT: link.exe {{.*}}-main1.a

; RUN: llvm-ar cru %t-main2.a %t.obj %t.dir/bitcode.obj
; RUN: lld-link %t.dir/bitcode.obj %t-main2.a /msvclto /out:%t.exe /opt:lldlto=1 /opt:icf \
; RUN:   /entry:main /verbose > %t.log || true
; RUN: FileCheck -check-prefix=OBJ %s < %t.log
; OBJ: link.exe {{.*}}-main2.a

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

declare void @foo()

define i32 @main() {
  call void @foo()
  ret i32 0
}
