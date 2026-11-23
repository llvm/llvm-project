// Check that clang emits the "target-abi" module flag for WebAssembly

// RUN: %clang_cc1 -triple wasm32-unknown-unknown -target-abi experimental-mv -emit-llvm -o - %s | FileCheck --check-prefix=MULTIVALUE %s
// RUN: %clang_cc1 -triple wasm64-unknown-unknown -target-abi experimental-mv -emit-llvm -o - %s | FileCheck --check-prefix=MULTIVALUE %s
// RUN: %clang_cc1 -triple wasm32-unknown-unknown -target-abi mvp -emit-llvm -o - %s | FileCheck --check-prefix=MVP %s

// No -target-abi: WebAssembly's default ABI is empty, so no flag is emitted.
// RUN: %clang_cc1 -triple wasm32-unknown-unknown -emit-llvm -o - %s | FileCheck --check-prefix=NONE %s

// MULTIVALUE: !{i32 1, !"target-abi", !"experimental-mv"}
// MVP: !{i32 1, !"target-abi", !"mvp"}
// NONE-NOT: !"target-abi"

int x;
