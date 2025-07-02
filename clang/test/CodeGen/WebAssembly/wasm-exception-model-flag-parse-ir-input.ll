; REQUIRES: webassembly-registered-target

; Check all the options parse
; RUN: %clang_cc1 -triple wasm32 -o - -emit-llvm -exception-model=none %s | FileCheck %s
; RUN: %clang_cc1 -triple wasm32 -o - -emit-llvm -exception-model=wasm %s | FileCheck %s
; RUN: %clang_cc1 -triple wasm32 -o - -emit-llvm -exception-model=dwarf %s | FileCheck %s
; RUN: %clang_cc1 -triple wasm32 -o - -emit-llvm -exception-model=sjlj %s | FileCheck %s

; RUN: not %clang_cc1 -triple wasm32 -o - -emit-llvm -exception-model=invalid %s 2>&1 | FileCheck -check-prefix=ERR %s

; CHECK-LABEL: define void @test(

; ERR: error: invalid value 'invalid' in '-exception-model=invalid'
define void @test() {
  ret void
}
