;; Test that a bitcode symbol defined in inline assembly (which wasm-ld
;; initially guesses is a FUNCTION) can be replaced by the LTO-generated
;; object symbol (which is correctly identified as a TAG) without error.

; RUN: llvm-as %s -o %t.o
; RUN: wasm-ld --export=foo %t.o -o %t.wasm
; RUN: obj2yaml %t.wasm | FileCheck %s

; CHECK:  - Type:            TAG
; CHECK:    TagTypes:        [ 1 ]
; CHECK:  - Name:            foo
; CHECK:    Kind:            TAG
; CHECK:    Index:           0

target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-unknown"

module asm ".globl foo"
module asm ".tagtype foo i32"
module asm "foo:"

define void @_start() {
  ret void
}
