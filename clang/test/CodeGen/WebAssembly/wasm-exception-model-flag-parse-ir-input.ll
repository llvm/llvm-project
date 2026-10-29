; REQUIRES: webassembly-registered-target

; Check all the options parse
; RUN: %clang_cc1 -triple wasm32 -o - -emit-llvm -exception-model=none %s | FileCheck -check-prefixes=CHECK,NONE %s
; RUN: %clang_cc1 -triple wasm32 -o - -emit-llvm -exception-model=wasm %s | FileCheck -check-prefixes=CHECK,WASM %s

; RUN: not %clang_cc1 -triple wasm32 -o - -emit-llvm -exception-model=invalid %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not %clang_cc1 -triple wasm32 -o - -emit-llvm -exception-model=dwarf %s 2>&1 | FileCheck -check-prefix=ERR-BE %s
; RUN: not %clang_cc1 -triple wasm32 -o - -emit-llvm -exception-model=sjlj %s 2>&1 | FileCheck -check-prefix=ERR-BE %s

; CHECK-LABEL: define void @test(

; The model is recorded on the module, as it is for source input.
; NONE: !{i32 1, !"exception-model", !"none"}
; WASM: !{i32 1, !"exception-model", !"wasm"}

; ERR: error: invalid value 'invalid' in '-exception-model=invalid'
; ERR-BE: fatal error: error in backend: -exception-model should be either 'none', 'wasm', or 'emscripten'
define void @test() {
  ret void
}
