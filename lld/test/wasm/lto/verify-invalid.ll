; RUN: llvm-as %s -o %t.o
; RUN: wasm-ld %t.o -o %t2 --lto-debug-pass-manager \
; RUN:   2>&1 | FileCheck -check-prefix=DEFAULT-NPM %s
; RUN: wasm-ld %t.o -o %t2 --lto-debug-pass-manager \
; RUN:   -disable-verify 2>&1 | FileCheck -check-prefix=DISABLE-NPM %s

target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

define void @_start() {
  ret void
}

; -disable-verify should disable the verification of bitcode.
; DEFAULT-NPM: Running pass: VerifierPass
; DEFAULT-NPM: Running pass: VerifierPass
; DEFAULT-NPM-NOT: Running pass: VerifierPass
; DISABLE-NPM-NOT: Running pass: VerifierPass
