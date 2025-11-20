; REQUIRES: x86
; RUN: mkdir -p %t.dir
; RUN: llvm-as -o %t.obj %s
; RUN: lld-link -out:%t.dll -dll -noentry %t.obj -export:test

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-msvc19.33.0"

$alt = comdat any

@alt = weak_odr dso_local global i32 0, comdat, align 4
@ext = external dso_local global i32, align 4

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @test() #0 {
entry:
  %0 = load i32, ptr @ext, align 4
  ret i32 %0
}

attributes #0 = { noinline nounwind optnone uwtable }

!llvm.linker.options = !{!0}

!0 = !{!"/alternatename:ext=alt"}
