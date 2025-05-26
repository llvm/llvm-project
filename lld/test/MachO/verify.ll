; REQUIRES: x86

; RUN: llvm-as %s -o %t.o
; RUN: %lld -dylib %t.o -o %t2 --lto-debug-pass-manager 2>&1 | FileCheck %s --check-prefix=VERIFY
; RUN: %lld -dylib %t.o -o %t2 --lto-debug-pass-manager --disable_verify 2>&1 | FileCheck %s --implicit-check-not=VerifierPass

target triple = "x86_64-apple-darwin"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @_start() {
entry:
  ret void
}

; VERIFY: Running pass: VerifierPass
; VERIFY: Running pass: VerifierPass
