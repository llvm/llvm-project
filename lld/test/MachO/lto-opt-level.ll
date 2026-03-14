; REQUIRES: x86

; RUN: rm -rf %t; mkdir %t
; RUN: llvm-as  %s -o %t/test.o

; RUN: %lld %t/test.o --lto-O0 -o %t/test
; RUN: llvm-nm -pa %t/test | FileCheck %s --check-prefixes=CHECK-O0

; RUN: %lld %t/test.o --lto-O2 -o %t/test
; RUN: llvm-nm -pa %t/test | FileCheck %s --check-prefixes=CHECK-O2

; RUN: %lld %t/test.o -o %t/test
; RUN: llvm-nm -pa %t/test | FileCheck %s --check-prefixes=CHECK-O2

; CHECK-O0: foo
; CHECK-O2-NOT: foo

target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define internal void @foo() {
  ret void
}

define void @main() {
  call void @foo()
  ret void
}
