; verify that errors in the LLVM backend during LTO manifest as lld
; errors

; RUN: llvm-as %s -o %t.o
; RUN: not wasm-ld --lto-O0 %t.o -o %t2 2>&1 | FileCheck %s

target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

define void @_start() {
  call ptr @foo()
  ret void
}

define ptr @foo() {
  %1 = call ptr @llvm.returnaddress(i32 0)
  ret ptr %1
}

declare ptr @llvm.returnaddress(i32)

; CHECK: error: {{.*}} WebAssembly hasn't implemented __builtin_return_address
