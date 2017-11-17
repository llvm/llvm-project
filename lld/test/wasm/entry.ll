; RUN: llc -filetype=obj %s -o %t.o
; RUN: lld -flavor wasm -e entry -o %t.wasm %t.o
; RUN: obj2yaml %t.wasm | FileCheck %s
; RUN: lld -flavor wasm --entry=entry -o %t.wasm %t.o
; RUN: obj2yaml %t.wasm | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown-wasm"

define hidden void @entry() local_unnamed_addr #0 {
entry:
  ret void
}

; CHECK:   - Type:            EXPORT
; CHECK:     Exports:         
; CHECK:       - Name:            memory
; CHECK:         Kind:            MEMORY
; CHECK:         Index:           0
; CHECK:       - Name:            entry
; CHECK:         Kind:            FUNCTION
; CHECK:         Index:           0
