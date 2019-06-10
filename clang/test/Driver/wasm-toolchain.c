// A basic clang -cc1 command-line. WebAssembly is somewhat special in
// enabling -fvisibility=hidden by default.

// RUN: %clang %s -### -no-canonical-prefixes -target wasm32-unknown-unknown 2>&1 \
// RUN:   | FileCheck -check-prefix=CC1 %s
// CC1: clang{{.*}} "-cc1" "-triple" "wasm32-unknown-unknown" {{.*}} "-fvisibility" "hidden" {{.*}}

// Ditto, but ensure that a user -fvisibility=default disables the default
// -fvisibility=hidden.

// RUN: %clang %s -### -target wasm32-unknown-unknown -fvisibility=default 2>&1 \
// RUN:   | FileCheck -check-prefix=FVISIBILITY_DEFAULT %s
// FVISIBILITY_DEFAULT-NOT: hidden

// A basic C link command-line with unknown OS.

// RUN: %clang -### -no-canonical-prefixes -target wasm32-unknown-unknown --sysroot=/foo %s 2>&1 \
// RUN:   | FileCheck -check-prefix=LINK %s
// LINK: clang{{.*}}" "-cc1" {{.*}} "-o" "[[temp:[^"]*]]"
// LINK: wasm-ld{{.*}}" "-L/foo/lib" "crt1.o" "[[temp]]" "-lc" "{{.*[/\\]}}libclang_rt.builtins-wasm32.a" "-o" "a.out"

// A basic C link command-line with optimization with unknown OS.

// RUN: %clang -### -O2 -no-canonical-prefixes -target wasm32-unknown-unknown --sysroot=/foo %s 2>&1 \
// RUN:   | FileCheck -check-prefix=LINK_OPT %s
// LINK_OPT: clang{{.*}}" "-cc1" {{.*}} "-o" "[[temp:[^"]*]]"
// LINK_OPT: wasm-ld{{.*}}" "-L/foo/lib" "crt1.o" "[[temp]]" "-lc" "{{.*[/\\]}}libclang_rt.builtins-wasm32.a" "-o" "a.out"

// A basic C link command-line with known OS.

// RUN: %clang -### -no-canonical-prefixes -target wasm32-wasi --sysroot=/foo %s 2>&1 \
// RUN:   | FileCheck -check-prefix=LINK_KNOWN %s
// LINK_KNOWN: clang{{.*}}" "-cc1" {{.*}} "-o" "[[temp:[^"]*]]"
// LINK_KNOWN: wasm-ld{{.*}}" "-L/foo/lib/wasm32-wasi" "crt1.o" "[[temp]]" "-lc" "{{.*[/\\]}}libclang_rt.builtins-wasm32.a" "-o" "a.out"

// A basic C link command-line with optimization with known OS.

// RUN: %clang -### -O2 -no-canonical-prefixes -target wasm32-wasi --sysroot=/foo %s 2>&1 \
// RUN:   | FileCheck -check-prefix=LINK_OPT_KNOWN %s
// LINK_OPT_KNOWN: clang{{.*}}" "-cc1" {{.*}} "-o" "[[temp:[^"]*]]"
// LINK_OPT_KNOWN: wasm-ld{{.*}}" "-L/foo/lib/wasm32-wasi" "crt1.o" "[[temp]]" "-lc" "{{.*[/\\]}}libclang_rt.builtins-wasm32.a" "-o" "a.out"

// A basic C compile command-line with known OS.

// RUN: %clang -### -no-canonical-prefixes -target wasm32-wasi --sysroot=/foo %s 2>&1 \
// RUN:   | FileCheck -check-prefix=COMPILE %s
// COMPILE: clang{{.*}}" "-cc1" {{.*}} "-internal-isystem" "/foo/include/wasm32-wasi" "-internal-isystem" "/foo/include"

// Thread-related command line tests.

// '-pthread' sets '-target-feature +atomics' and '--shared-memory'
// RUN: %clang -### -no-canonical-prefixes -target wasm32-unknown-unknown \
// RUN:    --sysroot=/foo %s -fuse-ld=wasm-ld -pthread 2>&1 \
// RUN:  | FileCheck -check-prefix=PTHREAD %s
// PTHREAD: clang{{.*}}" "-cc1" {{.*}} "-target-feature" "+atomics"
// PTHREAD: wasm-ld{{.*}}" "-lpthread" "--shared-memory"

// '-pthread' not allowed with '-mno-atomics'
// RUN: %clang -### -no-canonical-prefixes -target wasm32-unknown-unknown \
// RUN:     --sysroot=/foo %s -pthread -mno-atomics 2>&1 \
// RUN:   | FileCheck -check-prefix=PTHREAD_NO_ATOMICS %s
// PTHREAD_NO_ATOMICS: invalid argument '-pthread' not allowed with '-mno-atomics'
