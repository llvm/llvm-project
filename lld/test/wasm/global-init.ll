; RUN: llc -filetype=obj -mtriple=wasm32-unknown-unknown %s -o %t.o
; RUN: wasm-ld --no-entry --export=use --no-gc-sections %t.o -o %t.wasm
; RUN: obj2yaml %t.wasm | FileCheck %s

; The constant initializer of an address-space-1 Wasm global must survive
; linking, and the global's name must appear in the "name" custom section.

@gv = hidden addrspace(1) global i32 33554435
define i32 @use() {
  %v = load i32, ptr addrspace(1) @gv
  ret i32 %v
}

; CHECK:      - Type:            GLOBAL
; CHECK:          - Index:           1
; CHECK-NEXT:       Type:            I32
; CHECK-NEXT:       Mutable:         true
; CHECK-NEXT:       InitExpr:
; CHECK-NEXT:         Opcode:          I32_CONST
; CHECK-NEXT:         Value:           33554435

; CHECK:      - Type:            CUSTOM
; CHECK:        Name:            name
; CHECK:        GlobalNames:
; CHECK:          - Index:           1
; CHECK:            Name:            gv
