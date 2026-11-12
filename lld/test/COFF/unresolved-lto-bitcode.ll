; REQUIRES: x86
; RUN: rm -fr %t
; RUN: mkdir %t
; RUN: opt -thinlto-bc -o %t/main.obj %s
; RUN: opt -thinlto-bc -o %t/foo.obj %S/Inputs/lto-dep.ll
; RUN: not lld-link -lldsavetemps -out:%t/main.exe -entry:main \
; RUN:   -subsystem:console %t/main.obj %t/foo.obj 2>&1 | FileCheck %s
; RUN: ls %t | sort | FileCheck --check-prefix=FILE %s
; RUN: ls %t | count 2

; Check that the undefined symbol is reported, and that only the two
; object files we created are present in the directory (indicating that
; LTO did not run).
; CHECK: undefined symbol: bar
; CHECK: referenced by {{.*}}unresolved-lto-bitcode.ll
; CHECK: >>>           {{.*}}unresolved-lto-bitcode.ll.tmp/main.obj
; FILE: foo.obj
; FILE: main.obj

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define i32 @main() {
  call void @foo()
  call void @bar()
  ret i32 0
}

declare void @bar()
declare void @foo()
