; RUN: llvm-as %s -o %t.o
; RUN: wasm-ld %t.o -o %t.wasm --lto-O0

; Atomic operations will not fail to compile if atomics are not
; enabled because LLVM atomics will be lowered to regular ops.

target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-unknown-wasm"

@foo = hidden global i32 1

define void @_start() {
  %1 = load atomic i32, ptr @foo unordered, align 4
  ret void
}
