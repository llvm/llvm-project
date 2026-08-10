; REQUIRES: x86
; RUN: rm -rf %t; split-file %s %t
; RUN: llvm-as %t/archive.ll -o %t/archive.o
; RUN: llvm-ar rcs %t/archive.a  %t/archive.o
; RUN: llvm-as %t/obj.ll -o %t/obj.o

; RUN: %lld -dylib -lSystem %t/obj.o -load_hidden %t/archive.a -o %t/test.dylib
; RUN: llvm-nm %t/test.dylib | FileCheck %s
; CHECK: t _foo

;--- archive.ll
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @foo() noinline optnone {
    ret void
}

;--- obj.ll
target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

declare void @foo();

define void @main() {
  call void @foo()
  ret void
}
