; Test that LTO objects with no function can still be linked as shared
; libraries.
; We had a bug where the mutable-globals feature was not being added
; so the linker-generated import of `__stack_pointer` (which is currently
; mandatory for ; shared libraries) was generating a linker error.
; See https://bugs.llvm.org/show_bug.cgi?id=52339

; RUN: llvm-as %s -o %t.o
; RUN: wasm-ld -lto-O2 --experimental-pic -shared --no-gc-sections --export=tls_int %t.o -o %t.so
; RUN: obj2yaml %t.so  | FileCheck %s

target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-f128:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-emscripten"

@tls_int = dso_local thread_local global i32 99

; CHECK:  - Type:            CUSTOM
; CHECK:    Name:            target_features
; CHECK:    Features:
; CHECK:      - Prefix:          USED
; CHECK:        Name:            mutable-globals
