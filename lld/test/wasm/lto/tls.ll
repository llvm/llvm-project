; Test that LTO objects build with TLS and threading support can be linked into
; a single threaded binary.  Specifically the references to `__tls_base` that
; can be generated at LTO-time need to trigger the creation of the internal/fake
; `__tls_base` symbol in the linker.

; RUN: llvm-as %s -o %t1.o
; RUN: wasm-ld --export=tls_int --export=get_tls %t1.o -o %t
; RUN: obj2yaml %t | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-f128:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-emscripten"

@tls_int = dso_local thread_local global i32 99

define i32 @get_tls() #0 {
  %val = load i32, i32* @tls_int
  ret i32 %val
}

define void @_start() #0 {
  ret void
}

attributes #0 = { noinline nounwind optnone "target-features"="+atomics,+bulk-memory,+mutable-globals,+sign-ext" }

;      CHECK:   - Type:            GLOBAL
; CHECK-NEXT:     Globals:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Type:            I32
; CHECK-NEXT:         Mutable:         true
; CHECK-NEXT:         InitExpr:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           66576
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Type:            I32
; CHECK-NEXT:         Mutable:         false
; CHECK-NEXT:         InitExpr:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           1024

;      CHECK:     GlobalNames:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Name:            __stack_pointer
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Name:            __tls_base
