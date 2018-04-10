; RUN: llc -filetype=obj %s -o %t1.o
; RUN: llc -filetype=obj  %S/Inputs/custom.ll -o %t2.o
; RUN: wasm-ld --check-signatures --relocatable -o %t.wasm %t1.o %t2.o
; RUN: obj2yaml %t.wasm | FileCheck %s

target triple = "wasm32-unknown-unknown-wasm"

define i32 @_start() local_unnamed_addr {
entry:
  %retval = alloca i32, align 4
  ret i32 0
}

!0 = !{ !"red", !"extra" }
!wasm.custom_sections = !{ !0 }

; CHECK:        - Type:            CUSTOM
; CHECK-NEXT:     Name:            green
; CHECK-NEXT:     Payload:         '05677265656E626172717578'
; CHECK-NEXT:   - Type:            CUSTOM
; CHECK-NEXT:     Name:            red
; CHECK-NEXT:     Payload:         037265646578747261666F6F
