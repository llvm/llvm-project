; REQUIRES: x86
; RUN: opt -module-summary %s -o %t.o
; RUN: not %lld -lSystem -bitcode_bundle %t.o -o /dev/null 2>&1 | FileCheck %s
; CHECK: error: Option `-bitcode_bundle' is obsolete. Please modernize your usage.

target triple = "x86_64-apple-darwin"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @main() {
  ret void
}
