; RUN: llc -filetype=obj -wasm-enable-eh -exception-model=wasm -mattr=+exception-handling %p/Inputs/tag-section1.ll -o %t1.o
; RUN: llc -filetype=obj -wasm-enable-eh -exception-model=wasm -mattr=+exception-handling %p/Inputs/tag-section2.ll -o %t2.o
; RUN: llc -filetype=obj -wasm-enable-eh -exception-model=wasm -mattr=+exception-handling %s -o %t.o
; RUN: wasm-ld -o %t.wasm %t.o %t1.o %t2.o
; RUN: wasm-ld --export-all -o %t-export-all.wasm %t.o %t1.o %t2.o
; RUN: obj2yaml %t.wasm | FileCheck %s
; RUN: obj2yaml %t-export-all.wasm | FileCheck %s --check-prefix=EXPORT-ALL

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare void @foo(i8*)
declare void @bar(i8*)

define void @_start() {
  call void @foo(i8* null)
  call void @bar(i8* null)
  ret void
}

; CHECK:      Sections:
; CHECK-NEXT:   - Type:            TYPE
; CHECK-NEXT:     Signatures:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         ParamTypes:      []
; CHECK-NEXT:         ReturnTypes:     []
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         ParamTypes:
; CHECK-NEXT:           - I32
; CHECK-NEXT:         ReturnTypes:     []

; CHECK:        - Type:            TAG
; CHECK-NEXT:     TagTypes:        [ 1 ]

; Global section has to come after tag section
; CHECK:        - Type:            GLOBAL

; EXPORT-ALL:   - Type:            EXPORT
; EXPORT-ALL-NEXT Exports:
; EXPORT-ALL:       - Name:            __cpp_exception
; EXPORT-ALL:         Kind:            TAG
; EXPORT-ALL:         Index:           0
