; RUN: llc -filetype=obj %s -o %t.main.o
; RUN: lld -flavor wasm -o %t.wasm %t.main.o 2>&1 | FileCheck %s -check-prefix=CHECK-WARN
; RUN: not lld -flavor wasm --fatal-warnings -o %t.wasm %t.main.o 2>&1 | FileCheck %s -check-prefix=CHECK-FATAL

; CHECK-WARN: warning: Function type mismatch: _start
; CHECK-FATAL: error: Function type mismatch: _start

target triple = "wasm32-unknown-unknown-wasm"

define hidden i32 @_start(i32 %arg) local_unnamed_addr {
entry:
  ret i32 %arg
}

