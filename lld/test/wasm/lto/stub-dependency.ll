; RUN: llvm-as %s -o %t.bc
; RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %p/../Inputs/start.s -o %t.o
; RUN: echo "#STUB" > %t.so
; RUN: echo "A: B" >> %t.so
; RUN: wasm-ld %t.o %t.bc %t.so -o %t.wasm
; RUN: obj2yaml %t.wasm | FileCheck %s

; Test if LTO stub dependencies are preserved if a symbol they depend on is
; defined in bitcode and DCE'd and become undefined in the LTO process. Here 'B'
; should be preserved and exported.

target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-unknown"

define void @A() {
  ret void
}

define void @B() {
  ret void
}

; CHECK:      - Type:            EXPORT
; CHECK-NEXT:   Exports:
; CHECK-NEXT:     - Name:            memory
; CHECK-NEXT:       Kind:            MEMORY
; CHECK-NEXT:       Index:           0
; CHECK-NEXT:     - Name:            _start
; CHECK-NEXT:       Kind:            FUNCTION
; CHECK-NEXT:       Index:           0
; CHECK-NEXT:     - Name:            B
; CHECK-NEXT:       Kind:            FUNCTION
; CHECK-NEXT:       Index:           1
