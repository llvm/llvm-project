; RUN: llc -filetype=obj %s -o %t.o
; RUN: wasm-ld --allow-undefined -o %t.wasm %t.o
; RUN: obj2yaml %t.wasm | FileCheck %s

target triple = "wasm32-unknown-unknown-wasm"

define void @_start() {
  call void @foo();
  call void @qux();
  ret void
}

declare void @foo() #0
declare void @qux() #1

attributes #0 = { "wasm-import-module"="bar" }
attributes #1 = { "wasm-import-module"="" }

; CHECK:        - Type:            IMPORT
; CHECK-NEXT:     Imports:         
; CHECK-NEXT:       - Module:          bar
; CHECK-NEXT:         Field:           foo
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         SigIndex:        0
; CHECK-NEXT:       - Module:          ''
; CHECK-NEXT:         Field:           qux
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         SigIndex:        0
