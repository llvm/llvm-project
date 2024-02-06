// RUN: %clang --target=wasm32-unknown-unknown -### %s -fsyntax-only 2>&1 | FileCheck %s

// CHECK: "-fvisibility=hidden"

// RUN: %clang --target=wasm32-unknown-unknown -### %s 2>&1 | FileCheck %s -check-prefix=DEFAULT
// RUN: %clang --target=wasm32-unknown-unknown -### %s -mcpu=mvp 2>&1 | FileCheck %s -check-prefix=MVP
// RUN: %clang --target=wasm32-unknown-unknown -### %s -mcpu=bleeding-edge 2>&1 | FileCheck %s -check-prefix=BLEEDING-EDGE

// RUN: %clang --target=wasm32-unknown-unknown -### %s -mbulk-memory 2>&1 | FileCheck %s -check-prefix=BULK-MEMORY
// RUN: %clang --target=wasm32-unknown-unknown -### %s -mno-bulk-memory 2>&1 | FileCheck %s -check-prefix=NO-BULK-MEMORY

// BULK-MEMORY: "-target-feature" "+bulk-memory"
// NO-BULK-MEMORY: "-target-feature" "-bulk-memory"
// DEFAULT-NOT: "-target-feature" "-bulk-memory"
// MVP-NOT: "-target-feature" "+bulk-memory"
// BLEEDING-EDGE-NOT: "-target-feature" "-bulk-memory"

// RUN: %clang --target=wasm32-unknown-unknown -### %s -mmutable-globals 2>&1 | FileCheck %s -check-prefix=MUTABLE-GLOBALS
// RUN: %clang --target=wasm32-unknown-unknown -### %s -mno-mutable-globals 2>&1 | FileCheck %s -check-prefix=NO-MUTABLE-GLOBALS

// MUTABLE-GLOBALS: "-target-feature" "+mutable-globals"
// NO-MUTABLE-GLOBALS: "-target-feature" "-mutable-globals"
// DEFAULT-NOT: "-target-feature" "-mutable-globals"
// MVP-NOT: "-target-feature" "+mutable-globals"
// BLEEDING-EDGE-NOT: "-target-feature" "-mutable-globals"

// RUN: %clang --target=wasm32-unknown-unknown -### %s -msign-ext 2>&1 | FileCheck %s -check-prefix=SIGN-EXT
// RUN: %clang --target=wasm32-unknown-unknown -### %s -mno-sign-ext 2>&1 | FileCheck %s -check-prefix=NO-SIGN-EXT

// SIGN-EXT: "-target-feature" "+sign-ext"
// NO-SIGN-EXT: "-target-feature" "-sign-ext"
// DEFAULT-NOT: "-target-feature" "-sign-ext"
// MVP-NOT: "-target-feature" "+sign-ext"
// BLEEDING-EDGE-NOT: "-target-feature" "-sign-ext"

// RUN: %clang --target=wasm32-unknown-unknown -### %s -mnontrapping-fptoint 2>&1 | FileCheck %s -check-prefix=NONTRAPPING-FPTOINT
// RUN: %clang --target=wasm32-unknown-unknown -### %s -mno-nontrapping-fptoint 2>&1 | FileCheck %s -check-prefix=NO-NONTRAPPING-FPTOINT

// NONTRAPPING-FPTOINT: "-target-feature" "+nontrapping-fptoint"
// NO-NONTRAPPING-FPTOINT: "-target-feature" "-nontrapping-fptoint"
// DEFAULT-NOT: "-target-feature" "-nontrapping-fptoint"
// MVP-NOT: "-target-feature" "+nontrapping-fptoint"
// BLEEDING-EDGE-NOT: "-target-feature" "-nontrapping-fptoint"

// RUN: %clang --target=wasm32-unknown-unknown -### %s -mmultivalue 2>&1 | FileCheck %s -check-prefix=MULTIVALUE
// RUN: %clang --target=wasm32-unknown-unknown -### %s -mno-multivalue 2>&1 | FileCheck %s -check-prefix=NO-MULTIVALUE

// MULTIVALUE: "-target-feature" "+multivalue"
// NO-MULTIVALUE: "-target-feature" "-multivalue"
// DEFAULT-NOT: "-target-feature" "-multivalue"
// MVP-NOT: "-target-feature" "+multivalue"
// GENERIC-NOT: "-target-feature" "+multivalue"
// BLEEDING-EDGE-NOT: "-target-feature" "-multivalue"

// RUN: %clang --target=wasm32-unknown-unknown -### %s -mmultimemory 2>&1 | FileCheck %s -check-prefix=MULTIMEMORY
// RUN: %clang --target=wasm32-unknown-unknown -### %s -mno-multimemory 2>&1 | FileCheck %s -check-prefix=NO-MULTIMEMORY

// MULTIMEMORY: "-target-feature" "+multimemory"
// NO-MULTIMEMORY: "-target-feature" "-multimemory"
// DEFAULT-NOT: "-target-feature" "-multimemory"
// MVP-NOT: "-target-feature" "+multimemory"
// BLEEDING-EDGE-NOT: "-target-feature" "-multimemory"

// RUN: %clang --target=wasm32-unknown-unknown -### %s -matomics 2>&1 | FileCheck %s -check-prefix=ATOMICS
// RUN: %clang --target=wasm32-unknown-unknown -### %s -mno-atomics 2>&1 | FileCheck %s -check-prefix=NO-ATOMICS

// ATOMICS: "-target-feature" "+atomics"
// NO-ATOMICS: "-target-feature" "-atomics"
// DEFAULT-NOT: "-target-feature" "-atomics"
// MVP-NOT: "-target-feature" "+atomics"
// GENERIC-NOT: "-target-feature" "+atomics"
// BLEEDING-EDGE-NOT: "-target-feature" "-atomics"

// RUN: %clang --target=wasm32-unknown-unknown -### %s -mtail-call 2>&1 | FileCheck %s -check-prefix=TAIL-CALL
// RUN: %clang --target=wasm32-unknown-unknown -### %s -mno-tail-call 2>&1 | FileCheck %s -check-prefix=NO-TAIL-CALL

// TAIL-CALL: "-target-feature" "+tail-call"
// NO-TAIL-CALL: "-target-feature" "-tail-call"
// DEFAULT-NOT: "-target-feature" "-tail-call"
// MVP-NOT: "-target-feature" "+tail-call"
// GENERIC-NOT: "-target-feature" "+tail-call"
// BLEEDING-EDGE-NOT: "-target-feature" "-tail-call"

// RUN: %clang --target=wasm32-unknown-unknown -### %s -mreference-types 2>&1 | FileCheck %s -check-prefix=REFERENCE-TYPES
// RUN: %clang --target=wasm32-unknown-unknown -### %s -mno-reference-types 2>&1 | FileCheck %s -check-prefix=NO-REFERENCE-TYPES

// REFERENCE-TYPES: "-target-feature" "+reference-types"
// NO-REFERENCE-TYPES: "-target-feature" "-reference-types"
// DEFAULT-NOT: "-target-feature" "-reference-types"
// MVP-NOT: "-target-feature" "+reference-types"
// GENERIC-NOT: "-target-feature" "+reference-types"
// BLEEDING-EDGE-NOT: "-target-feature" "-reference-types"

// RUN: %clang --target=wasm32-unknown-unknown -### %s -msimd128 2>&1 | FileCheck %s -check-prefix=SIMD128
// RUN: %clang --target=wasm32-unknown-unknown -### %s -mno-simd128 2>&1 | FileCheck %s -check-prefix=NO-SIMD128

// SIMD128: "-target-feature" "+simd128"
// NO-SIMD128: "-target-feature" "-simd128"
// DEFAULT-NOT: "-target-feature" "-simd128"
// MVP-NOT: "-target-feature" "+simd128"
// GENERIC-NOT: "-target-feature" "+simd128"
// BLEEDING-EDGE-NOT: "-target-feature" "+simd128"

// RUN: %clang --target=wasm32-unknown-unknown -### %s -mrelaxed-simd 2>&1 | FileCheck %s -check-prefix=RELAXED-SIMD
// RUN: %clang --target=wasm32-unknown-unknown -### %s -mno-relaxed-simd 2>&1 | FileCheck %s -check-prefix=NO-RELAXED-SIMD

// RELAXED-SIMD: "-target-feature" "+relaxed-simd"
// NO-RELAXED-SIMD: "-target-feature" "-relaxed-simd"
// DEFAULT-NOT: "-target-feature" "-relaxed-simd"
// MVP-NOT: "-target-feature" "+relaxed-simd"
// GENERIC-NOT: "-target-feature" "+relaxed-simd"
// BLEEDING-EDGE-NOT: "-target-feature" "+relaxed-simd"

// RUN: %clang --target=wasm32-unknown-unknown -### %s -mexception-handling 2>&1 | FileCheck %s -check-prefix=EXCEPTION-HANDLING
// RUN: %clang --target=wasm32-unknown-unknown -### %s -mno-exception-handling 2>&1 | FileCheck %s -check-prefix=NO-EXCEPTION-HANDLING

// EXCEPTION-HANDLING: "-target-feature" "+exception-handling"
// NO-EXCEPTION-HANDLING: "-target-feature" "-exception-handling"
// DEFAULT-NOT: "-target-feature" "-exception-handling"
// MVP-NOT: "-target-feature" "+exception-handling"
// GENERIC-NOT: "-target-feature" "+exception-handling"
// BLEEDING-EDGE-NOT: "-target-feature" "+exception-handling"

// RUN: %clang --target=wasm32-unknown-unknown -### %s -mextended-const 2>&1 | FileCheck %s -check-prefix=EXTENDED-CONST
// RUN: %clang --target=wasm32-unknown-unknown -### %s -mno-extended-const 2>&1 | FileCheck %s -check-prefix=NO-EXTENDED-CONST

// EXTENDED-CONST: "-target-feature" "+extended-const"
// NO-EXTENDED-CONST: "-target-feature" "-extended-const"
// DEFAULT-NOT: "-target-feature" "-extended-const"
// MVP-NOT: "-target-feature" "+extended-const"
// GENERIC-NOT: "-target-feature" "+extended-const"
// BLEEDING-EDGE-NOT: "-target-feature" "+extended-const"
