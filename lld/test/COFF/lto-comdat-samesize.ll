; REQUIRES: x86

; RUN: split-file %s %t.dir
; RUN: llvm-as %t.dir/other.ll -o %t.other.bc
; RUN: llc -filetype=obj -o %t.main.obj %t.dir/main.ll

; RUN: lld-link -out:%t.exe -subsystem:console %t.other.bc %t.main.obj

#--- main.ll
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-msvc19.14.0"

$comdatData = comdat samesize

@comdatData = weak_odr dso_local global i32 42, comdat

define dso_local void @mainCRTStartup() {
entry:
  tail call void @other()
  ret void
}

declare dso_local void @other()

#--- other.ll
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-msvc19.14.0"

$comdatData = comdat samesize

@comdatData = weak_odr dso_local global i32 42, comdat

define dso_local void @other() {
entry:
  ret void
}
