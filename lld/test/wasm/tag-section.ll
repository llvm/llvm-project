; Static code
; RUN: llc -filetype=obj -wasm-enable-eh -exception-model=wasm -mattr=+exception-handling %p/Inputs/tag-section1.ll -o %t1.o
; RUN: llc -filetype=obj -wasm-enable-eh -exception-model=wasm -mattr=+exception-handling %p/Inputs/tag-section2.ll -o %t2.o
; RUN: llc -filetype=obj -wasm-enable-eh -exception-model=wasm -mattr=+exception-handling %s -o %t.o
; RUN: wasm-ld -o %t.wasm %t.o %t1.o %t2.o
; RUN: wasm-ld --export-all -o %t-export-all.wasm %t.o %t1.o %t2.o
; RUN: obj2yaml %t.wasm | FileCheck %s --check-prefix=NOPIC
; RUN: obj2yaml %t-export-all.wasm | FileCheck %s --check-prefix=NOPIC-EXPORT-ALL

; PIC code
; RUN: llc -filetype=obj -wasm-enable-eh -exception-model=wasm -mattr=+exception-handling -relocation-model=pic %p/Inputs/tag-section1.ll -o %t1.o
; RUN: llc -filetype=obj -wasm-enable-eh -exception-model=wasm -mattr=+exception-handling -relocation-model=pic %p/Inputs/tag-section2.ll -o %t2.o
; RUN: llc -filetype=obj -wasm-enable-eh -exception-model=wasm -mattr=+exception-handling -relocation-model=pic %s -o %t.o
; RUN: wasm-ld --import-undefined --experimental-pic -pie -o %t.wasm %t.o %t1.o %t2.o
; RUN: obj2yaml %t.wasm | FileCheck %s --check-prefix=PIC

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-emscripten"

declare void @foo(ptr)
declare void @bar(ptr)

define void @_start() {
  call void @foo(ptr null)
  call void @bar(ptr null)
  ret void
}

; NOPIC:      Sections:
; NOPIC-NEXT:   - Type:            TYPE
; NOPIC-NEXT:     Signatures:
; NOPIC-NEXT:       - Index:           0
; NOPIC-NEXT:         ParamTypes:      []
; NOPIC-NEXT:         ReturnTypes:     []
; NOPIC-NEXT:       - Index:           1
; NOPIC-NEXT:         ParamTypes:
; NOPIC-NEXT:           - I32
; NOPIC-NEXT:         ReturnTypes:     []

; NOPIC:        - Type:            TAG
; NOPIC-NEXT:     TagTypes:        [ 1 ]

; Global section has to come after tag section
; NOPIC:        - Type:            GLOBAL

; NOPIC-EXPORT-ALL:   - Type:            EXPORT
; NOPIC-EXPORT-ALL-NEXT Exports:
; NOPIC-EXPORT-ALL:       - Name:            __cpp_exception
; NOPIC-EXPORT-ALL:         Kind:            TAG
; NOPIC-EXPORT-ALL:         Index:           0

; In PIC mode, tags are undefined and imported from JS.
; PIC:        Sections:
; PIC:         - Type:            TYPE
; PIC-NEXT:      Signatures:
; PIC-NEXT:        - Index:           0
; PIC-NEXT:          ParamTypes:
; PIC-NEXT:            - I32
; PIC-NEXT:          ReturnTypes:     []
; PIC-NEXT:        - Index:           1
; PIC-NEXT:          ParamTypes:      []
; PIC-NEXT:          ReturnTypes:     []

; PIC:         - Type:            IMPORT
; PIC-NEXT:      Imports:
; PIC:             - Module:          env
; PIC:               Field:           __cpp_exception
; PIC-NEXT:          Kind:            TAG
; PIC-NEXT:          SigIndex:        0

; In PIC mode, tags should NOT be defined in the module; they are imported.
; PIC-NOT:     - Type:            TAG
