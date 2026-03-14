; REQUIRES: x86
; RUN: llvm-as %s -o %t.o
; RUN: env LLD_IN_TEST=1 ld.lld %t.o -o %t2 --lto-debug-pass-manager \
; RUN:   2>&1 | FileCheck -check-prefix=DEFAULT-NPM %s
; RUN: env LLD_IN_TEST=1 ld.lld %t.o -o %t2 --lto-debug-pass-manager \
; RUN:   -disable-verify 2>&1 | FileCheck -check-prefix=DISABLE-NPM %s
; RUN: env LLD_IN_TEST=1 ld.lld %t.o -o %t2 --lto-debug-pass-manager \
; RUN:   --plugin-opt=disable-verify 2>&1 | FileCheck -check-prefix=DISABLE-NPM %s

target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @_start() {
  ret void
}

; -disable-verify should disable the verification of bitcode.
; DEFAULT-NPM: Running pass: VerifierPass
; DEFAULT-NPM: Running pass: VerifierPass
; DEFAULT-NPM-NOT: Running pass: VerifierPass
; DISABLE-NPM-NOT: Running pass: VerifierPass
