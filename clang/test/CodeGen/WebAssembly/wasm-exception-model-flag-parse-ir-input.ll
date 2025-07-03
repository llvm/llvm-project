; REQUIRES: webassembly-registered-target

; Check all the options parse
; RUN: %clang_cc1 -triple wasm32 -o - -emit-llvm -exception-model=none %s | FileCheck %s
; RUN: %clang_cc1 -triple wasm32 -o - -emit-llvm -exception-model=wasm -mllvm -wasm-enable-eh %s | FileCheck %s

; RUN: not %clang_cc1 -triple wasm32 -o - -emit-llvm -exception-model=invalid %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not %clang_cc1 -triple wasm32 -o - -emit-llvm -exception-model=dwarf %s 2>&1 | FileCheck -check-prefix=ERR-BE %s
; RUN: not %clang_cc1 -triple wasm32 -o - -emit-llvm -exception-model=sjlj %s 2>&1 | FileCheck -check-prefix=ERR-BE %s

; CHECK-LABEL: define void @test(

; ERR: error: invalid value 'invalid' in '-exception-model=invalid'
; ERR-BE: fatal error: error in backend: -exception-model should be either 'none' or 'wasm'
define void @test() {
  ret void
}
