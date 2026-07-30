// A basic C link command-line with optimization with known OS and LTO enabled.

// RUN: %clang -### -O2 -flto --target=wasm32-wasi --sysroot=/foo %s 2>&1 \
// RUN:   | FileCheck -check-prefix=LINK_OPT_KNOWN %s
// LINK_OPT_KNOWN: "-cc1" {{.*}} "-o" "[[temp:[^"]*]]"
// LINK_OPT_KNOWN: wasm-ld{{.*}}" "-L/foo/lib/wasm32-wasi/llvm-lto/
