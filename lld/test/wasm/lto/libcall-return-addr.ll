; RUN: llvm-as %s -o %t.o
; RUN: llvm-as %p/Inputs/libcall-return-addr.ll -o %t.return-addr.o
; RUN: rm -f %t.a
; RUN: llvm-ar rcs %t.a %t.return-addr.o
; RUN: not wasm-ld --export-all %t.o %t.a -o %t.wasm 2>&1 | FileCheck %s

target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-i128:128-f128:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-emscripten"

@g_ptr = global ptr null

define void @_start() {
  %addr = call ptr @llvm.returnaddress(i32 1)
  store ptr %addr, ptr @g_ptr
  ret void
}

; CHECK: wasm-ld: error: {{.*}}return-addr.o): attempt to add bitcode file after LTO (emscripten_return_address)
