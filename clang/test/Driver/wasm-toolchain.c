// A basic clang -cc1 command-line. WebAssembly is somewhat special in
// enabling -ffunction-sections, -fdata-sections, and -fvisibility=hidden by
// default.

// RUN: %clang %s -### -no-canonical-prefixes -target wasm32-unknown-unknown 2>&1 | FileCheck -check-prefix=CC1 %s
// CC1: clang{{.*}} "-cc1" "-triple" "wasm32-unknown-unknown" {{.*}} "-fvisibility" "hidden" {{.*}} "-ffunction-sections" "-fdata-sections"

// Ditto, but ensure that a user -fno-function-sections disables the
// default -ffunction-sections.

// RUN: %clang %s -### -target wasm32-unknown-unknown -fno-function-sections 2>&1 | FileCheck -check-prefix=NO_FUNCTION_SECTIONS %s
// NO_FUNCTION_SECTIONS-NOT: function-sections

// Ditto, but ensure that a user -fno-data-sections disables the
// default -fdata-sections.

// RUN: %clang %s -### -target wasm32-unknown-unknown -fno-data-sections 2>&1 | FileCheck -check-prefix=NO_DATA_SECTIONS %s
// NO_DATA_SECTIONS-NOT: data-sections

// Ditto, but ensure that a user -fvisibility=default disables the default
// -fvisibility=hidden.

// RUN: %clang %s -### -target wasm32-unknown-unknown -fvisibility=default 2>&1 | FileCheck -check-prefix=FVISIBILITY_DEFAULT %s
// FVISIBILITY_DEFAULT-NOT: hidden

// A basic C link command-line.

// RUN: %clang -### -no-canonical-prefixes -target wasm32-unknown-unknown --sysroot=/foo %s 2>&1 | FileCheck -check-prefix=LINK %s
// LINK: clang{{.*}}" "-cc1" {{.*}} "-o" "[[temp:[^"]*]]"
// LINK: lld{{.*}}" "-flavor" "wasm" "-L/foo/lib" "[[temp]]" "-allow-undefined-file" "wasm.syms" "-lc" "-lcompiler_rt" "-o" "a.out"

// A basic C link command-line with optimization.

// RUN: %clang -### -O2 -no-canonical-prefixes -target wasm32-unknown-unknown --sysroot=/foo %s 2>&1 | FileCheck -check-prefix=LINK_OPT %s
// LINK_OPT: clang{{.*}}" "-cc1" {{.*}} "-o" "[[temp:[^"]*]]"
// LINK_OPT: lld{{.*}}" "-flavor" "wasm" "-L/foo/lib" "[[temp]]" "-allow-undefined-file" "wasm.syms" "-lc" "-lcompiler_rt" "-o" "a.out"
