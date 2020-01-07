; Optional linker-synthetic symbols are only created if they are undefined
; in the final output.
; This test is for a regression where an explicit --export of an lazy archive
; symbol caused an undefined referece to an optional symbol to occur *after*
; the optional symbols were created.

; RUN: llc -filetype=obj %s -o %t.o
; RUN: llc -filetype=obj %S/Inputs/optional-symbol.ll -o %t.a1.o
; RUN: rm -f %t.a
; RUN: llvm-ar rcs %t.a %t.a1.o
; RUN: wasm-ld --export=get_optional %t.o %t.a -o %t.wasm
; RUN: obj2yaml %t.wasm | FileCheck %s

target triple = "wasm32-unknown-unknown"

define void @_start() {
entry:
  ret void
}

; CHECK:      FunctionNames:
; CHECK-NEXT:   - Index:           0
; CHECK-NEXT:     Name:            _start
; CHECK-NEXT:   - Index:           1
; CHECK-NEXT:     Name:            get_optional
