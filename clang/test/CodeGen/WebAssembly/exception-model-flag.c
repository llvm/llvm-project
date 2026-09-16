// REQUIRES: webassembly-registered-target

// Verify clang records the "exception-model" module flag for the WebAssembly
// exception models. The target-independent models are covered in
// exception-model-flag.c.

// Wasm EH (needs the backend enable flag) records the "wasm" model.
// RUN: %clang_cc1 -triple wasm32-unknown-unknown -fexceptions -exception-model=wasm -mllvm -wasm-enable-eh -emit-llvm %s -o - | FileCheck %s --check-prefix=WASM

// Emscripten EH records the "emscripten" model.
// RUN: %clang_cc1 -triple wasm32-unknown-emscripten -fexceptions -exception-model=emscripten -emit-llvm %s -o - | FileCheck %s --check-prefix=EMSCRIPTEN

void f(void) {}

// WASM: !{i32 1, !"exception-model", !"wasm"}
// EMSCRIPTEN: !{i32 1, !"exception-model", !"emscripten"}
