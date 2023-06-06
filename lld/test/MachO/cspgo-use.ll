; REQUIRES: x86

; Create an empty profile
; RUN: echo > %t.proftext
; RUN: llvm-profdata merge %t.proftext -o %t.profdata

; RUN: llvm-as %s -o %t.o
; RUN: %lld -dylib --cs-profile-path=%t.profdata %t.o -o %t --lto-debug-pass-manager 2>&1 | FileCheck %s --implicit-check-not=PGOInstrumentation

; CHECK: Running pass: PGOInstrumentationUse

target triple = "x86_64-apple-darwin"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @foo() {
entry:
  ret void
}
