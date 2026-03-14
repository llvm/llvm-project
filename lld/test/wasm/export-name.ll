; RUN: llc -filetype=obj %s -o %t.o
; RUN: wasm-ld -o %t.wasm %t.o
; RUN: obj2yaml %t.wasm | FileCheck %s

target triple = "wasm32-unknown-unknown"

define void @foo() #0 {
    ret void
}

define void @qux() #1 {
    ret void
}

define void @_start() {
    call void @foo()
    ret void
}

attributes #0 = { "wasm-export-name"="bar" }

attributes #1 = { "wasm-export-name"="" }

; CHECK:       - Type:            EXPORT
; CHECK-NEXT:    Exports:
; CHECK-NEXT:      - Name:            memory
; CHECK-NEXT:        Kind:            MEMORY
; CHECK-NEXT:        Index:           0
; CHECK-NEXT:      - Name:            bar
; CHECK-NEXT:        Kind:            FUNCTION
; CHECK-NEXT:        Index:           0
; CHECK-NEXT:      - Name:            ''
; CHECK-NEXT:        Kind:            FUNCTION
; CHECK-NEXT:        Index:           1
; CHECK-NEXT:      - Name:            _start
; CHECK-NEXT:        Kind:            FUNCTION
; CHECK-NEXT:        Index:           2
