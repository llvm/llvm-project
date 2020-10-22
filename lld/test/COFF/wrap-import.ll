// REQUIRES: x86

// Check that wrapping works when the wrapped symbol is imported from a
// different DLL.

// RUN: split-file %s %t.dir
// RUN: llc %t.dir/main.ll -o %t.main.obj --filetype=obj
// RUN: llvm-as %t.dir/main.ll -o %t.main.bc
// RUN: llvm-mc -filetype=obj -triple=x86_64-win32-gnu %t.dir/lib.s -o %t.lib.obj

// RUN: lld-link -dll -out:%t.lib.dll %t.lib.obj -noentry -export:func -implib:%t.lib.lib
// RUN: lld-link -out:%t.exe %t.main.obj %t.lib.lib -entry:entry -subsystem:console -wrap:func
// RUN: lld-link -out:%t.exe %t.main.bc %t.lib.lib -entry:entry -subsystem:console -wrap:func

#--- main.ll
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-w64-windows-gnu"

declare void @func()

define void @entry() {
  call void @func()
  ret void
}

declare void @__real_func()

define void @__wrap_func() {
  call void @__real_func()
  ret void
}

#--- lib.s
.global func
func:
  ret
