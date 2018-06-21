; RUN: llc -O0 -filetype=obj %s -o %t.o

; RUN: wasm-ld -o %t.wasm %t.o
; RUN: obj2yaml %t.wasm | FileCheck %s

; RUN: wasm-ld --export-all -o %t.wasm %t.o
; RUN: obj2yaml %t.wasm | FileCheck %s -check-prefix=EXPORT

; RUN: wasm-ld --export-all --no-gc-sections -o %t.wasm %t.o
; RUN: obj2yaml %t.wasm | FileCheck %s -check-prefix=NOGC

; Verify the --export-all flag exports hidden symbols

target triple = "wasm32-unknown-unknown"

define hidden void @bar() local_unnamed_addr {
entry:
  ret void
}

define hidden void @foo() local_unnamed_addr {
entry:
  ret void
}

define hidden void @_start() local_unnamed_addr {
entry:
  call void @foo()
  ret void
}

; CHECK:      - Type:            EXPORT
; CHECK:         - Name:            _start
; CHECK-NOT:     - Name:            bar
; CHECK-NOT:     - Name:            foo

; EXPORT:     - Type:            EXPORT
; EXPORT:        - Name:            _start
; EXPORT-NOT:    - Name:            bar
; EXPORT:        - Name:            foo

; NOGC:       - Type:            EXPORT
; NOGC:        - Name:            _start
; NOGC:        - Name:            bar
; NOGC:        - Name:            foo
