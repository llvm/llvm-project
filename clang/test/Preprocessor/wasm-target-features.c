// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm32-unknown-unknown -msimd128 \
// RUN:   | FileCheck %s -check-prefix=SIMD128
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm64-unknown-unknown -msimd128 \
// RUN:   | FileCheck %s -check-prefix=SIMD128
//
// SIMD128:#define __wasm_simd128__ 1{{$}}
//
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm32-unknown-unknown -munimplemented-simd128 \
// RUN:   | FileCheck %s -check-prefix=SIMD128-UNIMPLEMENTED
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm64-unknown-unknown -munimplemented-simd128 \
// RUN:   | FileCheck %s -check-prefix=SIMD128-UNIMPLEMENTED
//
// SIMD128-UNIMPLEMENTED:#define __wasm_unimplemented_simd128__ 1{{$}}
//
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm32-unknown-unknown -mcpu=mvp \
// RUN:   | FileCheck %s -check-prefix=MVP
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm64-unknown-unknown -mcpu=mvp \
// RUN:   | FileCheck %s -check-prefix=MVP
//
// MVP-NOT:#define __wasm_simd128__
//
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm32-unknown-unknown -mcpu=bleeding-edge \
// RUN:   | FileCheck %s -check-prefix=BLEEDING_EDGE
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm64-unknown-unknown -mcpu=bleeding-edge \
// RUN:   | FileCheck %s -check-prefix=BLEEDING_EDGE
//
// BLEEDING_EDGE:#define __wasm_simd128__ 1{{$}}
//
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm32-unknown-unknown -mcpu=bleeding-edge -mno-simd128 \
// RUN:   | FileCheck %s -check-prefix=BLEEDING_EDGE_NO_SIMD128
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm64-unknown-unknown -mcpu=bleeding-edge -mno-simd128 \
// RUN:   | FileCheck %s -check-prefix=BLEEDING_EDGE_NO_SIMD128
//
// BLEEDING_EDGE_NO_SIMD128-NOT:#define __wasm_simd128__
