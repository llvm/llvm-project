; RUN: llvm-as %s -o %t.o
; RUN: wasm-ld %t.o -o %t.wasm
; RUN: obj2yaml %t.wasm | FileCheck %s

; Test that undefined weak external functions are handled in the LTO case
; We had a bug where stub function generation was failing because functions
; that are in bitcode (pre-LTO) don't have signatures assigned.

target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare extern_weak i32 @foo()

declare extern_weak i32 @bar()

; The reference to bar here will exist in the LTO file, but the LTO process will
; remove it, so it will never be referenced by an normal object file, and
; therefore never have a signature.
define void @unused_function() #0 {
entry:
    %call2 = call i32 @bar()
    ret void
}

define void @_start() #0 {
entry:
    %call2 = call i32 @foo()
    ret void
}

; CHECK: Name:            'undefined_weak:foo'
