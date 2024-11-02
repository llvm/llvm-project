; REQUIRES: x86
; RUN: rm -f %t.a
; RUN: llvm-as -o %t.o %s
; RUN: llvm-as -o %t2.o %S/Inputs/libcall-archive.ll
; RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux -o %t3.o %S/Inputs/libcall-archive.s
; RUN: llvm-ar rcs %t.a %t2.o %t3.o
; RUN: ld.lld -o %t %t.o %t.a
; RUN: llvm-nm %t | FileCheck %s
; RUN: ld.lld -o %t2 %t.o --start-lib %t2.o %t3.o --end-lib
; RUN: llvm-nm %t2 | FileCheck %s

; CHECK-NOT: T __sync_val_compare_and_swap_8
; CHECK: T _start
; CHECK: T memcpy

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @_start(ptr %a, ptr %b) {
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr %a, ptr %b, i64 1024, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1)
