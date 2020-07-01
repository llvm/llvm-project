; RUN: llvm-as %s -o %t.o
; RUN: llvm-as %p/Inputs/libcall-truncsfhf2.ll -o %t.truncsfhf2.o
; RUN: rm -f %t.a
; RUN: llvm-ar rcs %t.a %t.truncsfhf2.o
; RUN: not wasm-ld --export-all %t.o %t.a -o %t.wasm 2>&1 | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

@g_float = global float 0.0
@g_half = global half 0.0

define void @_start() {
  %val1 = load float, float* @g_float
  %v0 = fptrunc float %val1 to half
  store half %v0, half* @g_half
  ret void
}

; CHECK: wasm-ld: error: {{.*}}truncsfhf2.o): attempt to add bitcode file after LTO.
