; REQUIRES: x86
; RUN: llvm-as -o %t.obj %s

; Verify that an invalid but unused input pdb file doesn't trigger any warning or error.
; RUN: lld-link /out:%t.exe %t.obj %S/Inputs/bad-block-size.pdb /entry:main /subsystem:console /WX

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define void @main() {
  ret void
}


