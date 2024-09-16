// RUN: %clang --target=wasm32-unknown-unknown -### %s -fsyntax-only 2>&1 | FileCheck %s

// CHECK: "-fvisibility=hidden"

// RUN: %clang --target=wasm32-unknown-unknown -### %s -mcpu=mvp 2>&1 | FileCheck %s -check-prefix=MVP
// RUN: %clang --target=wasm32-unknown-unknown -### %s 2>&1 | FileCheck %s -check-prefix=GENERIC
// RUN: %clang --target=wasm32-unknown-unknown -### %s -mcpu=generic 2>&1 | FileCheck %s -check-prefix=GENERIC
// RUN: %clang --target=wasm32-unknown-unknown -### %s -mcpu=bleeding-edge 2>&1 | FileCheck %s -check-prefix=BLEEDING-EDGE

// MVP: "-target-cpu" "mvp"
// GENERIC: "-target-cpu" "generic"
// BLEEDING-EDGE: "-target-cpu" "bleeding-edge"

// RUN: %clang --target=wasm32-unknown-unknown -### %s -matomics 2>&1 | FileCheck %s -check-prefix=ATOMICS
// RUN: %clang --target=wasm32-unknown-unknown -### %s -mno-atomics 2>&1 | FileCheck %s -check-prefix=NO-ATOMICS

// ATOMICS: "-target-feature" "+atomics"
// NO-ATOMICS: "-target-feature" "-atomics"

// RUN: %clang --target=wasm32-unknown-unknown -### %s -mbulk-memory 2>&1 | FileCheck %s -check-prefix=BULK-MEMORY
// RUN: %clang --target=wasm32-unknown-unknown -### %s -mno-bulk-memory 2>&1 | FileCheck %s -check-prefix=NO-BULK-MEMORY

// BULK-MEMORY: "-target-feature" "+bulk-memory"
// NO-BULK-MEMORY: "-target-feature" "-bulk-memory"

// RUN: %clang --target=wasm32-unknown-unknown -### %s -mexception-handling 2>&1 | FileCheck %s -check-prefix=EXCEPTION-HANDLING
// RUN: %clang --target=wasm32-unknown-unknown -### %s -mno-exception-handling 2>&1 | FileCheck %s -check-prefix=NO-EXCEPTION-HANDLING

// EXCEPTION-HANDLING: "-target-feature" "+exception-handling"
// NO-EXCEPTION-HANDLING: "-target-feature" "-exception-handling"

// RUN: %clang --target=wasm32-unknown-unknown -### %s -mextended-const 2>&1 | FileCheck %s -check-prefix=EXTENDED-CONST
// RUN: %clang --target=wasm32-unknown-unknown -### %s -mno-extended-const 2>&1 | FileCheck %s -check-prefix=NO-EXTENDED-CONST

// EXTENDED-CONST: "-target-feature" "+extended-const"
// NO-EXTENDED-CONST: "-target-feature" "-extended-const"

// RUN: %clang --target=wasm32-unknown-unknown -### %s -mfp16 2>&1 | FileCheck %s -check-prefix=HALF-PRECISION
// RUN: %clang --target=wasm32-unknown-unknown -### %s -mno-fp16 2>&1 | FileCheck %s -check-prefix=NO-HALF-PRECISION

// HALF-PRECISION: "-target-feature" "+fp16"
// NO-HALF-PRECISION: "-target-feature" "-fp16"

// RUN: %clang --target=wasm32-unknown-unknown -### %s -mmultimemory 2>&1 | FileCheck %s -check-prefix=MULTIMEMORY
// RUN: %clang --target=wasm32-unknown-unknown -### %s -mno-multimemory 2>&1 | FileCheck %s -check-prefix=NO-MULTIMEMORY

// MULTIMEMORY: "-target-feature" "+multimemory"
// NO-MULTIMEMORY: "-target-feature" "-multimemory"

// RUN: %clang --target=wasm32-unknown-unknown -### %s -mmultivalue 2>&1 | FileCheck %s -check-prefix=MULTIVALUE
// RUN: %clang --target=wasm32-unknown-unknown -### %s -mno-multivalue 2>&1 | FileCheck %s -check-prefix=NO-MULTIVALUE

// MULTIVALUE: "-target-feature" "+multivalue"
// NO-MULTIVALUE: "-target-feature" "-multivalue"

// RUN: %clang --target=wasm32-unknown-unknown -### %s -mmutable-globals 2>&1 | FileCheck %s -check-prefix=MUTABLE-GLOBALS
// RUN: %clang --target=wasm32-unknown-unknown -### %s -mno-mutable-globals 2>&1 | FileCheck %s -check-prefix=NO-MUTABLE-GLOBALS

// MUTABLE-GLOBALS: "-target-feature" "+mutable-globals"
// NO-MUTABLE-GLOBALS: "-target-feature" "-mutable-globals"

// RUN: %clang --target=wasm32-unknown-unknown -### %s -mnontrapping-fptoint 2>&1 | FileCheck %s -check-prefix=NONTRAPPING-FPTOINT
// RUN: %clang --target=wasm32-unknown-unknown -### %s -mno-nontrapping-fptoint 2>&1 | FileCheck %s -check-prefix=NO-NONTRAPPING-FPTOINT

// NONTRAPPING-FPTOINT: "-target-feature" "+nontrapping-fptoint"
// NO-NONTRAPPING-FPTOINT: "-target-feature" "-nontrapping-fptoint"

// RUN: %clang --target=wasm32-unknown-unknown -### %s -mreference-types 2>&1 | FileCheck %s -check-prefix=REFERENCE-TYPES
// RUN: %clang --target=wasm32-unknown-unknown -### %s -mno-reference-types 2>&1 | FileCheck %s -check-prefix=NO-REFERENCE-TYPES

// REFERENCE-TYPES: "-target-feature" "+reference-types"
// NO-REFERENCE-TYPES: "-target-feature" "-reference-types"

// RUN: %clang --target=wasm32-unknown-unknown -### %s -mrelaxed-simd 2>&1 | FileCheck %s -check-prefix=RELAXED-SIMD
// RUN: %clang --target=wasm32-unknown-unknown -### %s -mno-relaxed-simd 2>&1 | FileCheck %s -check-prefix=NO-RELAXED-SIMD

// RELAXED-SIMD: "-target-feature" "+relaxed-simd"
// NO-RELAXED-SIMD: "-target-feature" "-relaxed-simd"

// RUN: %clang --target=wasm32-unknown-unknown -### %s -msign-ext 2>&1 | FileCheck %s -check-prefix=SIGN-EXT
// RUN: %clang --target=wasm32-unknown-unknown -### %s -mno-sign-ext 2>&1 | FileCheck %s -check-prefix=NO-SIGN-EXT

// SIGN-EXT: "-target-feature" "+sign-ext"
// NO-SIGN-EXT: "-target-feature" "-sign-ext"

// RUN: %clang --target=wasm32-unknown-unknown -### %s -msimd128 2>&1 | FileCheck %s -check-prefix=SIMD128
// RUN: %clang --target=wasm32-unknown-unknown -### %s -mno-simd128 2>&1 | FileCheck %s -check-prefix=NO-SIMD128

// SIMD128: "-target-feature" "+simd128"
// NO-SIMD128: "-target-feature" "-simd128"

// RUN: %clang --target=wasm32-unknown-unknown -### %s -mtail-call 2>&1 | FileCheck %s -check-prefix=TAIL-CALL
// RUN: %clang --target=wasm32-unknown-unknown -### %s -mno-tail-call 2>&1 | FileCheck %s -check-prefix=NO-TAIL-CALL

// TAIL-CALL: "-target-feature" "+tail-call"
// NO-TAIL-CALL: "-target-feature" "-tail-call"
