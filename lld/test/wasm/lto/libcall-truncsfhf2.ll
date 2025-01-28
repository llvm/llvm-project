; RUN: llvm-as %s -o %t.o
; RUN: llvm-as %p/Inputs/libcall-truncsfhf2.ll -o %t.truncsfhf2.o
; RUN: rm -f %t.a
; RUN: llvm-ar rcs %t.a %t.truncsfhf2.o
; RUN: not wasm-ld --export-all %t.o %t.a -o %t.wasm 2>&1 | FileCheck %s

target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-unknown"

@g_float = global float 0.0
@g_half = global half 0.0

define void @_start() {
  %val1 = load float, ptr @g_float
  %v0 = fptrunc float %val1 to half
  store half %v0, ptr @g_half
  ret void
}

; CHECK: wasm-ld: error: {{.*}}truncsfhf2.o): attempt to add bitcode file after LTO (__truncsfhf2)
