; RUN: llc -filetype=obj %s -o %t.o
; RUN: wasm-ld -o %t.wasm %t.o
; RUN: obj2yaml %t.wasm | FileCheck %s

target triple = "wasm32-unknown-unknown"

declare void @f0() #0
declare void @f1() #1

define void @_start() {
    call void @f0()
    call void @f1()
    ret void
}

attributes #0 = { "wasm-import-module"="somewhere" "wasm-import-name"="something" }
attributes #1 = { "wasm-import-module"="otherwhere" "wasm-import-name"="" }

; CHECK:        - Type:            IMPORT
; CHECK-NEXT:     Imports:
; CHECK-NEXT:       - Module:          somewhere
; CHECK-NEXT:         Field:           something
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         SigIndex:        0
; CHECK-NEXT:       - Module:          otherwhere
; CHECK-NEXT:         Field:           ''
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         SigIndex:        0

; CHECK:        - Type:            CUSTOM
; CHECK-NEXT:     Name:            name
; CHECK-NEXT:     FunctionNames:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Name:            f0
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Name:            f1
