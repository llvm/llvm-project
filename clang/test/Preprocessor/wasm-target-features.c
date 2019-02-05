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
// RUN:     -target wasm32-unknown-unknown -mnontrapping-fptoint \
// RUN:   | FileCheck %s -check-prefix=NONTRAPPING-FPTOINT
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm64-unknown-unknown -mnontrapping-fptoint \
// RUN:   | FileCheck %s -check-prefix=NONTRAPPING-FPTOINT
//
// NONTRAPPING-FPTOINT:#define __wasm_nontrapping_fptoint__ 1{{$}}
//
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm32-unknown-unknown -msign-ext \
// RUN:   | FileCheck %s -check-prefix=SIGN-EXT
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm64-unknown-unknown -msign-ext \
// RUN:   | FileCheck %s -check-prefix=SIGN-EXT
//
// SIGN-EXT:#define __wasm_sign_ext__ 1{{$}}
//
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm32-unknown-unknown -mexception-handling \
// RUN:   | FileCheck %s -check-prefix=EXCEPTION-HANDLING
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm64-unknown-unknown -mexception-handling \
// RUN:   | FileCheck %s -check-prefix=EXCEPTION-HANDLING
//
// EXCEPTION-HANDLING:#define __wasm_exception_handling__ 1{{$}}
//
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm32-unknown-unknown -mbulk-memory \
// RUN:   | FileCheck %s -check-prefix=BULK-MEMORY
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm64-unknown-unknown -mbulk-memory \
// RUN:   | FileCheck %s -check-prefix=BULK-MEMORY
//
// BULK-MEMORY:#define __wasm_bulk_memory__ 1{{$}}
//
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm32-unknown-unknown -mcpu=mvp \
// RUN:   | FileCheck %s -check-prefix=MVP
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm64-unknown-unknown -mcpu=mvp \
// RUN:   | FileCheck %s -check-prefix=MVP
//
// MVP-NOT:#define __wasm_simd128__
// MVP-NOT:#define __wasm_unimplemented_simd128__
// MVP-NOT:#define __wasm_nontrapping_fptoint__
// MVP-NOT:#define __wasm_sign_ext__
// MVP-NOT:#define __wasm_exception_handling__
// MVP-NOT:#define __wasm_bulk_memory__
//
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm32-unknown-unknown -mcpu=bleeding-edge \
// RUN:   | FileCheck %s -check-prefix=BLEEDING-EDGE
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm64-unknown-unknown -mcpu=bleeding-edge \
// RUN:   | FileCheck %s -check-prefix=BLEEDING-EDGE
//
// BLEEDING-EDGE:#define __wasm_nontrapping_fptoint__ 1{{$}}
// BLEEDING-EDGE:#define __wasm_sign_ext__ 1{{$}}
// BLEEDING-EDGE:#define __wasm_simd128__ 1{{$}}
// BLEEDING-EDGE-NOT:#define __wasm_unimplemented_simd128__ 1{{$}}
//
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm32-unknown-unknown -mcpu=bleeding-edge -mno-simd128 \
// RUN:   | FileCheck %s -check-prefix=BLEEDING-EDGE-NO-SIMD128
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm64-unknown-unknown -mcpu=bleeding-edge -mno-simd128 \
// RUN:   | FileCheck %s -check-prefix=BLEEDING-EDGE-NO-SIMD128
//
// BLEEDING-EDGE-NO-SIMD128-NOT:#define __wasm_simd128__
