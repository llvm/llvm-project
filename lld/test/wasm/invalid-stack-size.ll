; RUN: llc -filetype=obj %s -o %t.o
; RUN: not lld -flavor wasm -o %t.wasm -z stack-size=1 %t.o 2>&1 | FileCheck %s

target triple = "wasm32-unknown-unknown-wasm"

define i32 @_start() local_unnamed_addr #1 {
entry:
  ret i32 0
}

; CHECK: error: stack size must be 16-byte aligned
