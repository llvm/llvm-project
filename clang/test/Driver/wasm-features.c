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
