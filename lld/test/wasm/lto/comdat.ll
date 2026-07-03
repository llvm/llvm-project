; Verify that comdat symbols can be defined in LTO objects.  We had a
; regression where the comdat handling code was causing symbol in the lto object
; to be ignored.
; RUN: llvm-as %s -o %t.bc
; RUN: llc -filetype=obj %s -o %t.o
; RUN: wasm-ld %t.bc %t.o -o %t.wasm
; RUN: wasm-ld %t.o %t.bc -o %t.wasm

target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-unknown"

$foo = comdat any

define void @_start() comdat($foo) {
entry:
  ret void
}
