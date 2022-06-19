; RUN: llvm-as -o %t.bc %s
; RUN: wasm-ld --no-lto-legacy-pass-manager --lto-debug-pass-manager -o /dev/null %t.bc 2>&1 | FileCheck %s

; CHECK: Running pass: GlobalOptPass

target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

define void @_start() local_unnamed_addr {
entry:
  ret void
}
