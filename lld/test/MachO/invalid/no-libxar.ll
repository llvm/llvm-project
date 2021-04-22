; REQUIRES: x86
; UNSUPPORTED: xar
; RUN: opt -module-summary %s -o %t.o
; RUN: not %lld -lSystem -bitcode_bundle %t.o -o /dev/null 2>&1 | FileCheck %s
; CHECK: error: -bitcode_bundle unsupported because LLD wasn't built with libxar

target triple = "x86_64-apple-darwin"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @main() {
  ret void
}
