; REQUIRES: x86
; RUN: split-file %s %t.dir

; RUN: llvm-as %t.dir/main.ll -o %t.main.bc
; RUN: llvm-as %t.dir/other.ll -o %t.other.bc
; RUN: llc %t.dir/main.ll -o %t.main.obj --filetype=obj
; RUN: llc %t.dir/other.ll -o %t.other.obj --filetype=obj

; RUN: lld-link %t.main.obj %t.other.obj -entry:main -out:%t.exe
; RUN: lld-link %t.main.bc  %t.other.bc  -entry:main -out:%t.exe
; RUN: lld-link %t.main.bc  %t.other.obj -entry:main -out:%t.exe
; RUN: lld-link %t.main.obj %t.other.bc  -entry:main -out:%t.exe

;--- main.ll
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.11.0"

define dso_local i32 @main() local_unnamed_addr {
entry:
  %call = tail call i32 @foo()
  %0 = load i32, ptr @variable, align 4
  %add = add nsw i32 %0, %call
  ret i32 %add
}

@variable = external dllimport local_unnamed_addr global i32, align 4

declare dllimport i32 @foo() local_unnamed_addr

!llvm.module.flags = !{!1}

!1 = !{i32 1, !"ThinLTO", i32 0}

;--- other.ll
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.11.0"

define dso_local i32 @foo() local_unnamed_addr {
entry:
  ret i32 42
}

@variable = dso_local local_unnamed_addr global i32 1, align 4

!llvm.module.flags = !{!1}

!1 = !{i32 1, !"ThinLTO", i32 0}
