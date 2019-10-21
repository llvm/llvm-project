; RUN: llc -filetype=obj %s -o %t.o
; RUN: wasm-ld -no-gc-sections --no-entry %t.o -o %t.wasm
; RUN: obj2yaml %t.wasm | FileCheck %s

; Test that the data section is skipped entirely when there are only
; bss segments

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

@a = global [1000 x i8] zeroinitializer, align 1
@b = global i32 0

; CHECK-NOT: - Type:            DATA
