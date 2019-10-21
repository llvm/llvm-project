; RUN: llvm-as %s -o %t.o
; RUN: wasm-ld --allow-undefined -o %t.wasm %t.o
; RUN: obj2yaml %t.wasm | FileCheck %s

target triple = "wasm32-unknown-unknown-wasm"
target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"

define void @_start() {
  call void @foo();
  ret void
}

declare void @foo() #0

attributes #0 = { "wasm-import-module"="bar" "wasm-import-name"="customfoo" }

; CHECK:        - Type:            IMPORT
; CHECK-NEXT:     Imports:
; CHECK-NEXT:       - Module:          bar
; CHECK-NEXT:         Field:           customfoo
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         SigIndex:        0
