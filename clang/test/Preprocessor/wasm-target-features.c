// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm32-unknown-unknown -matomics \
// RUN:   | FileCheck %s -check-prefix=ATOMICS
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm64-unknown-unknown -matomics \
// RUN:   | FileCheck %s -check-prefix=ATOMICS
//
// ATOMICS: #define __wasm_atomics__ 1{{$}}

// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm32-unknown-unknown -pthread \
// RUN:   | FileCheck %s -check-prefix=PTHREAD
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm64-unknown-unknown -pthread \
// RUN:   | FileCheck %s -check-prefix=PTHREAD
//
// PTHREAD: #define __wasm_atomics__ 1{{$}}

// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm32-unknown-unknown -mbulk-memory \
// RUN:   | FileCheck %s -check-prefix=BULK-MEMORY
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm64-unknown-unknown -mbulk-memory \
// RUN:   | FileCheck %s -check-prefix=BULK-MEMORY
//
// BULK-MEMORY: #define __wasm_bulk_memory__ 1{{$}}

// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm32-unknown-unknown -mexception-handling \
// RUN:   | FileCheck %s -check-prefix=EXCEPTION-HANDLING
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm64-unknown-unknown -mexception-handling \
// RUN:   | FileCheck %s -check-prefix=EXCEPTION-HANDLING
//
// EXCEPTION-HANDLING: #define __wasm_exception_handling__ 1{{$}}

// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm32-unknown-unknown -mextended-const \
// RUN:   | FileCheck %s -check-prefix=EXTENDED-CONST
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm64-unknown-unknown -mextended-const \
// RUN:   | FileCheck %s -check-prefix=EXTENDED-CONST
//
// EXTENDED-CONST: #define __wasm_extended_const__ 1{{$}}

// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm32-unknown-unknown -mhalf-precision \
// RUN:   | FileCheck %s -check-prefix=HALF-PRECISION
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm64-unknown-unknown -mhalf-precision \
// RUN:   | FileCheck %s -check-prefix=HALF-PRECISION
//
// HALF-PRECISION: #define __wasm_half_precision__ 1{{$}}

// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm32-unknown-unknown -mmultimemory \
// RUN:   | FileCheck %s -check-prefix=MULTIMEMORY
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm64-unknown-unknown -mmultimemory \
// RUN:   | FileCheck %s -check-prefix=MULTIMEMORY
//
// MULTIMEMORY: #define __wasm_multimemory__ 1{{$}}

// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm32-unknown-unknown -mmultivalue \
// RUN:   | FileCheck %s -check-prefix=MULTIVALUE
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm64-unknown-unknown -mmultivalue \
// RUN:   | FileCheck %s -check-prefix=MULTIVALUE
//
// MULTIVALUE: #define __wasm_multivalue__ 1{{$}}

// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm32-unknown-unknown -mmutable-globals \
// RUN:   | FileCheck %s -check-prefix=MUTABLE-GLOBALS
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm64-unknown-unknown -mmutable-globals \
// RUN:   | FileCheck %s -check-prefix=MUTABLE-GLOBALS
//
// MUTABLE-GLOBALS: #define __wasm_mutable_globals__ 1{{$}}

// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm32-unknown-unknown -mnontrapping-fptoint \
// RUN:   | FileCheck %s -check-prefix=NONTRAPPING-FPTOINT
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm64-unknown-unknown -mnontrapping-fptoint \
// RUN:   | FileCheck %s -check-prefix=NONTRAPPING-FPTOINT
//
// NONTRAPPING-FPTOINT: #define __wasm_nontrapping_fptoint__ 1{{$}}

// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm32-unknown-unknown -mreference-types \
// RUN:   | FileCheck %s -check-prefix=REFERENCE-TYPES
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm64-unknown-unknown -mreference-types \
// RUN:   | FileCheck %s -check-prefix=REFERENCE-TYPES
//
// REFERENCE-TYPES: #define __wasm_reference_types__ 1{{$}}

// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm32-unknown-unknown -mrelaxed-simd \
// RUN:   | FileCheck %s -check-prefix=RELAXED-SIMD
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm64-unknown-unknown -mrelaxed-simd \
// RUN:   | FileCheck %s -check-prefix=RELAXED-SIMD
//
// RELAXED-SIMD: #define __wasm_relaxed_simd__ 1{{$}}

// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm32-unknown-unknown -msign-ext \
// RUN:   | FileCheck %s -check-prefix=SIGN-EXT
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm64-unknown-unknown -msign-ext \
// RUN:   | FileCheck %s -check-prefix=SIGN-EXT
//
// SIGN-EXT: #define __wasm_sign_ext__ 1{{$}}

// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm32-unknown-unknown -msimd128 \
// RUN:   | FileCheck %s -check-prefix=SIMD128
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm64-unknown-unknown -msimd128 \
// RUN:   | FileCheck %s -check-prefix=SIMD128
//
// SIMD128: #define __wasm_simd128__ 1{{$}}

// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm32-unknown-unknown -mtail-call \
// RUN:   | FileCheck %s -check-prefix=TAIL-CALL
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm64-unknown-unknown -mtail-call \
// RUN:   | FileCheck %s -check-prefix=TAIL-CALL
//
// TAIL-CALL: #define __wasm_tail_call__ 1{{$}}

// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm32-unknown-unknown -mcpu=mvp \
// RUN:   | FileCheck %s -check-prefix=MVP
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm64-unknown-unknown -mcpu=mvp \
// RUN:   | FileCheck %s -check-prefix=MVP
//
// MVP-NOT: #define __wasm_atomics__ 1{{$}}
// MVP-NOT: #define __wasm_bulk_memory__ 1{{$}}
// MVP-NOT: #define __wasm_exception_handling__ 1{{$}}
// MVP-NOT: #define __wasm_extended_const__ 1{{$}}
// MVP-NOT: #define __wasm_half_precision__ 1{{$}}
// MVP-NOT: #define __wasm_multimemory__ 1{{$}}
// MVP-NOT: #define __wasm_multivalue__ 1{{$}}
// MVP-NOT: #define __wasm_mutable_globals__ 1{{$}}
// MVP-NOT: #define __wasm_nontrapping_fptoint__ 1{{$}}
// MVP-NOT: #define __wasm_reference_types__ 1{{$}}
// MVP-NOT: #define __wasm_relaxed_simd__ 1{{$}}
// MVP-NOT: #define __wasm_sign_ext__ 1{{$}}
// MVP-NOT: #define __wasm_simd128__ 1{{$}}
// MVP-NOT: #define __wasm_tail_call__ 1{{$}}

// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm32-unknown-unknown -mcpu=generic \
// RUN:   | FileCheck %s -check-prefix=GENERIC-INCLUDE
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm64-unknown-unknown -mcpu=generic \
// RUN:   | FileCheck %s -check-prefix=GENERIC-INCLUDE
//
// GENERIC-INCLUDE-DAG: #define __wasm_multivalue__ 1{{$}}
// GENERIC-INCLUDE-DAG: #define __wasm_mutable_globals__ 1{{$}}
// GENERIC-INCLUDE-DAG: #define __wasm_reference_types__ 1{{$}}
// GENERIC-INCLUDE-DAG: #define __wasm_sign_ext__ 1{{$}}
//
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm32-unknown-unknown -mcpu=generic \
// RUN:   | FileCheck %s -check-prefix=GENERIC
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm64-unknown-unknown -mcpu=generic \
// RUN:   | FileCheck %s -check-prefix=GENERIC
//
// GENERIC-NOT: #define __wasm_atomics__ 1{{$}}
// GENERIC-NOT: #define __wasm_bulk_memory__ 1{{$}}
// GENERIC-NOT: #define __wasm_exception_handling__ 1{{$}}
// GENERIC-NOT: #define __wasm_extended_const__ 1{{$}}
// GENERIC-NOT: #define __wasm_half_precision__ 1{{$}}
// GENERIC-NOT: #define __wasm_multimemory__ 1{{$}}
// GENERIC-NOT: #define __wasm_nontrapping_fptoint__ 1{{$}}
// GENERIC-NOT: #define __wasm_relaxed_simd__ 1{{$}}
// GENERIC-NOT: #define __wasm_simd128__ 1{{$}}
// GENERIC-NOT: #define __wasm_tail_call__ 1{{$}}

// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm32-unknown-unknown -mcpu=bleeding-edge \
// RUN:   | FileCheck %s -check-prefix=BLEEDING-EDGE-INCLUDE
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm64-unknown-unknown -mcpu=bleeding-edge \
// RUN:   | FileCheck %s -check-prefix=BLEEDING-EDGE-INCLUDE
//
// BLEEDING-EDGE-INCLUDE-DAG: #define __wasm_atomics__ 1{{$}}
// BLEEDING-EDGE-INCLUDE-DAG: #define __wasm_bulk_memory__ 1{{$}}
// BLEEDING-EDGE-INCLUDE-DAG: #define __wasm_exception_handling__ 1{{$}}
// BLEEDING-EDGE-INCLUDE-DAG: #define __wasm_extended_const__ 1{{$}}
// BLEEDING-EDGE-INCLUDE-DAG: #define __wasm_half_precision__ 1{{$}}
// BLEEDING-EDGE-INCLUDE-DAG: #define __wasm_multimemory__ 1{{$}}
// BLEEDING-EDGE-INCLUDE-DAG: #define __wasm_multivalue__ 1{{$}}
// BLEEDING-EDGE-INCLUDE-DAG: #define __wasm_mutable_globals__ 1{{$}}
// BLEEDING-EDGE-INCLUDE-DAG: #define __wasm_nontrapping_fptoint__ 1{{$}}
// BLEEDING-EDGE-INCLUDE-DAG: #define __wasm_reference_types__ 1{{$}}
// BLEEDING-EDGE-INCLUDE-DAG: #define __wasm_relaxed_simd__ 1{{$}}
// BLEEDING-EDGE-INCLUDE-DAG: #define __wasm_sign_ext__ 1{{$}}
// BLEEDING-EDGE-INCLUDE-DAG: #define __wasm_simd128__ 1{{$}}
// BLEEDING-EDGE-INCLUDE-DAG: #define __wasm_tail_call__ 1{{$}}

// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm32-unknown-unknown -mcpu=bleeding-edge -mno-simd128 \
// RUN:   | FileCheck %s -check-prefix=BLEEDING-EDGE-NO-SIMD128
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target wasm64-unknown-unknown -mcpu=bleeding-edge -mno-simd128 \
// RUN:   | FileCheck %s -check-prefix=BLEEDING-EDGE-NO-SIMD128
//
// BLEEDING-EDGE-NO-SIMD128-NOT: #define __wasm_simd128__ 1{{$}}
