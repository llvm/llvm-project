; REQUIRES: x86
; RUN: opt -module-summary %s -o %t.o
; RUN: not %lld -dylib -lSystem %t.o -o /dev/null 2>&1 | FileCheck %s
; CHECK: error: _foo has protected visibility, which is not supported by Mach-O

target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define protected void @foo() {
  ret void
}
