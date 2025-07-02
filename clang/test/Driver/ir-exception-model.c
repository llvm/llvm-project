// RUN: %clang -### -target wasm32-unknown-unknown -fwasm-exceptions -c -S -o - %S/Inputs/file.ll 2>&1 | FileCheck %s
// RUN: %clang -### -target wasm32-unknown-unknown -Xclang -exception-model=wasm -c -S -o - %S/Inputs/file.ll 2>&1 | FileCheck %s
// RUN: %clang -### -target wasm32-unknown-unknown -Xclang -exception-model=dwarf -c -S -o - %S/Inputs/file.ll 2>&1 | FileCheck -check-prefix=DWARF %s
// RUN: %clang -### -target wasm32-unknown-unknown -Xclang -exception-model=sjlj -c -S -o - %S/Inputs/file.ll 2>&1 | FileCheck -check-prefix=SJLJ %s
// RUN: %clang -### -target wasm32-unknown-unknown -Xclang -exception-model=wineh -c -S -o - %S/Inputs/file.ll 2>&1 | FileCheck -check-prefix=WINEH %s
// RUN: %clang -### -target wasm32-unknown-unknown -Xclang -exception-model=arst -c -S -o - %S/Inputs/file.ll 2>&1 | FileCheck -check-prefix=INVALID %s

// Check that -fwasm-exceptions propagates -exception-model to cc1

// CHECK: "-exception-model=wasm"
// DWARF: "-exception-model=dwarf"
// SJLJ: "-exception-model=sjlj"
// WINEH: "-exception-model=wineh"
// INVALID: "-exception-model=arst"
