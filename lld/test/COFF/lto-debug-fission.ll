; REQUIRES: x86

; RUN: opt %s -o %t1.o
; RUN: rm -rf %t.dir

; Test to ensure that -dwodir:$DIR creates .dwo files under $DIR
; RUN: lld-link -dwodir:%t.dir -noentry -dll %t1.o -out:%t.dll
; RUN: llvm-readobj -h %t.dir/0.dwo | FileCheck %s

; CHECK: Format: COFF-x86-64

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-msvc"

define void @f() {
entry:
  ret void
}
