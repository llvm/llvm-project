; RUN: llvm-as %s -o %t.o
; RUN: wasm-ld %t.o %t.o -o %t.wasm -r
; RUN: llvm-readobj --symbols %t.wasm | FileCheck %s

target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-unknown-wasm"

define weak void @f() {
  ret void
}

; CHECK:        Symbol {
; CHECK-NEXT:     Name: f
; CHECK-NEXT:     Type: FUNCTION (0x0)
; CHECK-NEXT:     Flags [ (0x1)
; CHECK-NEXT:       BINDING_WEAK (0x1)
; CHECK-NEXT:     ]
; CHECK-NEXT:     ElementIndex: 0x0
; CHECK-NEXT:   }
