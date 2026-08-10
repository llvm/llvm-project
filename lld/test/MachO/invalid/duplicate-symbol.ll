; REQUIRES: x86
; RUN: rm -rf %t; mkdir -p %t
; RUN: opt -module-summary %s -o %t/first.o
; RUN: opt -module-summary %s -o %t/second.o
; RUN: not %lld -dylib -lSystem %t/first.o %t/second.o -o /dev/null 2>&1 | FileCheck %s
; CHECK:      error: duplicate symbol: _foo
; CHECK-NEXT: >>> defined in {{.*}}/first.o
; CHECK-NEXT: >>> defined in {{.*}}/second.o

target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @foo() {
  ret void
}
