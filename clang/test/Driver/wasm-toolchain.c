// A basic clang -cc1 command-line. WebAssembly is somewhat special in
// enabling -fvisibility=hidden by default.

// RUN: %clang %s -### -no-canonical-prefixes -target wasm32-unknown-unknown 2>&1 | FileCheck -check-prefix=CC1 %s
// CC1: clang{{.*}} "-cc1" "-triple" "wasm32-unknown-unknown" {{.*}} "-fvisibility" "hidden" {{.*}}

// Ditto, but ensure that a user -fvisibility=default disables the default
// -fvisibility=hidden.

// RUN: %clang %s -### -target wasm32-unknown-unknown -fvisibility=default 2>&1 | FileCheck -check-prefix=FVISIBILITY_DEFAULT %s
// FVISIBILITY_DEFAULT-NOT: hidden

// A basic C link command-line.

// RUN: %clang -### -no-canonical-prefixes -target wasm32-unknown-unknown --sysroot=/foo -fuse-ld=wasm-ld %s 2>&1 | FileCheck -check-prefix=LINK %s
// LINK: clang{{.*}}" "-cc1" {{.*}} "-o" "[[temp:[^"]*]]"
// LINK: wasm-ld{{.*}}" "-L/foo/lib" "crt1.o" "[[temp]]" "-lc" "{{.*[/\\]}}libclang_rt.builtins-wasm32.a" "-o" "a.out"

// A basic C link command-line with optimization.

// RUN: %clang -### -O2 -no-canonical-prefixes -target wasm32-unknown-unknown --sysroot=/foo -fuse-ld=wasm-ld %s 2>&1 | FileCheck -check-prefix=LINK_OPT %s
// LINK_OPT: clang{{.*}}" "-cc1" {{.*}} "-o" "[[temp:[^"]*]]"
// LINK_OPT: wasm-ld{{.*}}" "-L/foo/lib" "crt1.o" "[[temp]]" "-lc" "{{.*[/\\]}}libclang_rt.builtins-wasm32.a" "-o" "a.out"
