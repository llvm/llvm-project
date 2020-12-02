; REQUIRES: x86
; RUN: llvm-as %s -o %t.o
; RUN: ld.lld %t.o -o %t2 -mllvm -debug-pass=Arguments --no-lto-new-pass-manager \
; RUN:   2>&1 | FileCheck -check-prefix=DEFAULT-LPM %s
; RUN: ld.lld %t.o -o %t2 -mllvm -debug-pass=Arguments --no-lto-new-pass-manager \
; RUN:   -disable-verify 2>&1 | FileCheck -check-prefix=DISABLE-LPM %s
; RUN: ld.lld %t.o -o %t2 -mllvm -debug-pass=Arguments --no-lto-new-pass-manager \
; RUN:   --plugin-opt=disable-verify 2>&1 | FileCheck -check-prefix=DISABLE-LPM %s
; RUN: ld.lld %t.o -o %t2 --lto-new-pass-manager --lto-debug-pass-manager \
; RUN:   2>&1 | FileCheck -check-prefix=DEFAULT-NPM %s
; RUN: ld.lld %t.o -o %t2 --lto-new-pass-manager --lto-debug-pass-manager \
; RUN:   -disable-verify 2>&1 | FileCheck -check-prefix=DISABLE-NPM %s
; RUN: ld.lld %t.o -o %t2 --lto-new-pass-manager --lto-debug-pass-manager \
; RUN:   --plugin-opt=disable-verify 2>&1 | FileCheck -check-prefix=DISABLE-NPM %s

target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @_start() {
  ret void
}

; -disable-verify should disable the verification of bitcode.
; DEFAULT-LPM:     Pass Arguments: {{.*}} -verify {{.*}} -verify
; DISABLE-LPM-NOT: Pass Arguments: {{.*}} -verify {{.*}} -verify
; DEFAULT-NPM: Running pass: VerifierPass
; DEFAULT-NPM: Running pass: VerifierPass
; DEFAULT-NPM-NOT: Running pass: VerifierPass
; DISABLE-NPM-NOT: Running pass: VerifierPass
