// REQUIRES: llvm-driver, lld, webassembly-registered-target

// The multicall driver registers Clang and LLD in one LLVMToolSession. Verify
// that Clang discovers wasm-ld in that session rather than spawning it.
// RUN: split-file %s %t
// RUN: %clang --target=wasm32-unknown-unknown -nostdlib -fuse-ld=lld \
// RUN:   -Wl,--no-entry -### %t/add.c %t/sub.c 2>&1 \
// RUN:   | FileCheck %s --check-prefix=PRINT
// PRINT: {{.*}}clang{{.*}}-cc1
// PRINT: {{.*}}clang{{.*}}-cc1
// PRINT: (in-process)
// PRINT-NEXT: {{.*}}wasm-ld

// Also exercise multiple integrated cc1 jobs followed by the in-process link.
// RUN: %clang --target=wasm32-unknown-unknown -nostdlib -fuse-ld=lld \
// RUN:   -Wl,--no-entry -Wl,--export=add -Wl,--export=sub \
// RUN:   %t/add.c %t/sub.c -o %t.wasm
// RUN: llvm-readobj --file-headers %t.wasm | FileCheck %s --check-prefix=WASM
// WASM: Format: WASM

//--- add.c
int add(int lhs, int rhs) { return lhs + rhs; }

//--- sub.c
int sub(int lhs, int rhs) { return lhs - rhs; }
