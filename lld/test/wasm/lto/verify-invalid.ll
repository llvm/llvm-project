; RUN: llvm-as %s -o %t.o
; RUN: wasm-ld %t.o -o %t2 --no-lto-new-pass-manager -mllvm -debug-pass=Arguments \
; RUN:   2>&1 | FileCheck -check-prefix=DEFAULT-LPM %s
; RUN: wasm-ld %t.o -o %t2 --no-lto-new-pass-manager -mllvm -debug-pass=Arguments \
; RUN:   -disable-verify 2>&1 | FileCheck -check-prefix=DISABLE-LPM %s
; RUN: wasm-ld %t.o -o %t2 --lto-new-pass-manager --lto-debug-pass-manager \
; RUN:   2>&1 | FileCheck -check-prefix=DEFAULT-NPM %s
; RUN: wasm-ld %t.o -o %t2 --lto-new-pass-manager --lto-debug-pass-manager \
; RUN:   -disable-verify 2>&1 | FileCheck -check-prefix=DISABLE-NPM %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

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
