; REQUIRES: x86

; RUN: llvm-as %s -o %t.o
; RUN: %lld -dylib --cs-profile-generate --cs-profile-path=default_%m.profraw %t.o -o %t --lto-debug-pass-manager 2>&1 | FileCheck %s --implicit-check-not=PGOInstrumentation

; CHECK: PGOInstrumentationGen

target triple = "x86_64-apple-darwin"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

@__llvm_profile_runtime = global i32 0, align 4

define void @foo() {
entry:
  ret void
}
