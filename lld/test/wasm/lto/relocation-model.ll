;; The explicit relocation model flag.

; RUN: llvm-as %s -o %t.o

; RUN: wasm-ld %t.o -o %t.wasm -save-temps -r -mllvm -relocation-model=pic
; RUN: llvm-readobj -r %t.wasm.lto.o | FileCheck %s --check-prefix=PIC

; RUN: wasm-ld %t.o -o %t_static.wasm -save-temps -r -mllvm -relocation-model=static
; RUN: llvm-readobj -r %t_static.wasm.lto.o | FileCheck %s --check-prefix=STATIC

; PIC: R_WASM_GLOBAL_INDEX_LEB foo
; STATIC: R_WASM_MEMORY_ADDR_LEB foo

target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-unknown"

@foo = external global i32
define i32 @_start() {
  %t = load i32, i32* @foo
  ret i32 %t
}
