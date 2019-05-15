; Verify that comdat symbols can be defined in LTO objects.  We had a
; regression where the comdat handling code was causing symbol in the lto object
; to be ignored.
; RUN: llvm-as %s -o %t.o
; RUN: wasm-ld %t.o %t.o -o %t.wasm

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

$foo = comdat any

define linkonce_odr void @_start() comdat($foo) {
entry:
  ret void
}
