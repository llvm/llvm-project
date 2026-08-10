; RUN: llvm-as %s -o %t.o
; RUN: wasm-ld %t.o -o %t.wasm
; RUN: obj2yaml %t.wasm | FileCheck %s

; CHECK:        - Type:            CUSTOM
; CHECK-NEXT:     Name:            name
; CHECK-NEXT:     FunctionNames:   
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Name:            _start

target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-unknown-wasm"

define void @_start() {
  ret void
}
