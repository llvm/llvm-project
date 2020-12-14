; RUN: llvm-as -o %t.bc %s
; RUN: wasm-ld --no-lto-legacy-pass-manager --lto-debug-pass-manager -o /dev/null %t.bc 2>&1 | FileCheck %s
; RUN: wasm-ld --no-lto-legacy-pass-manager --lto-debug-pass-manager --lto-legacy-pass-manager -o /dev/null %t.bc 2>&1 | FileCheck %s --allow-empty --check-prefix=LPM

; CHECK: Starting llvm::Module pass manager run
; CHECK: Finished llvm::Module pass manager run
; LPM-NOT: Starting llvm::Module pass manager run

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

define void @_start() local_unnamed_addr {
entry:
  ret void
}
