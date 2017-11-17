; RUN: llc -filetype=obj %s -o %t.o
; RUN: lld -flavor wasm -o %t.wasm %t.o
; RUN: llvm-readobj -file-headers %t.wasm | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown-wasm"

define hidden void @_start() local_unnamed_addr #0 {
entry:
    ret void
}

; CHECK: Format: WASM
; CHECK: Arch: wasm32
; CHECK: AddressSize: 32bit
; CHECK: Version: 0x1
