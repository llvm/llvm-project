; REQUIRES: webassembly-registered-target

; An -exception-model= matching the input module's flag is accepted.
; RUN: %clang_cc1 -triple wasm32 -o - -emit-llvm -exception-model=wasm %s | FileCheck %s

; A contradicting one is rejected.
; RUN: not %clang_cc1 -triple wasm32 -o - -emit-llvm -exception-model=none %s 2>&1 | FileCheck -check-prefix=NONE %s
; RUN: not %clang_cc1 -triple wasm32 -o - -emit-llvm -exception-model=emscripten %s 2>&1 | FileCheck -check-prefix=EM %s

; CHECK: !{i32 1, !"exception-model", !"wasm"}

; NONE: error: '-exception-model=none' does not match the module's exception model 'wasm'
; EM: error: '-exception-model=emscripten' does not match the module's exception model 'wasm'

define void @test() {
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"exception-model", !"wasm"}
