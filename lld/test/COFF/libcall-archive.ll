; REQUIRES: x86
; RUN: rm -f %t.a
; RUN: llvm-as -o %t.obj %s
; RUN: llvm-as -o %t2.obj %S/Inputs/libcall-archive.ll
; RUN: llvm-mc -filetype=obj -triple=i686-unknown-windows -o %t3.obj %S/Inputs/libcall-archive.s
; RUN: llvm-ar rcs %t.a %t2.obj %t3.obj
; RUN: lld-link -out:%t.exe -subsystem:console -entry:start -safeseh:no -lldmap:- %t.obj %t.a | FileCheck %s

; CHECK-NOT: ___sync_val_compare_and_swap_8
; CHECK: _start
; CHECK: _memcpy

target datalayout = "e-m:x-p:32:32-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-unknown-windows"

define void @start(ptr %a, ptr %b) {
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr %a, ptr %b, i64 1024, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1)
