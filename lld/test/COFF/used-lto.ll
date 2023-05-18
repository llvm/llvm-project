; REQUIRES: x86
; RUN: llvm-as -o %t.obj %s
; RUN: lld-link -dll -debug -opt:ref -noentry -out:%t.dll %t.obj
; RUN: llvm-pdbutil dump -publics %t.pdb | FileCheck %s

; CHECK: S_PUB32 {{.*}} `foo`

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

@llvm.used = appending global [1 x ptr] [ptr @foo], section "llvm.metadata"

define void @foo() {
  ret void
}
