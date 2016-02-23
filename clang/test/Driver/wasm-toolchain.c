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
// LINK: lld{{.*}}" "-flavor" "ld" "-L/foo/lib32" "crt1.o" "crti.o" "[[temp]]" "-lc" "-lcompiler_rt" "crtn.o" "-o" "a.out"

// A basic C link command-line with optimization. WebAssembly is somewhat
// special in enabling --gc-sections by default.

// RUN: %clang -### -O2 -no-canonical-prefixes -target wasm32-unknown-unknown --sysroot=/foo %s 2>&1 | FileCheck -check-prefix=LINK_OPT %s
// LINK_OPT: clang{{.*}}" "-cc1" {{.*}} "-o" "[[temp:[^"]*]]"
// LINK_OPT: lld{{.*}}" "-flavor" "ld" "--gc-sections" "-L/foo/lib32" "crt1.o" "crti.o" "[[temp]]" "-lc" "-lcompiler_rt" "crtn.o" "-o" "a.out"

// Ditto, but ensure that a user --no-gc-sections comes after the
// default --gc-sections.

// RUN: %clang -### -O2 -no-canonical-prefixes -target wasm32-unknown-unknown --sysroot=/foo -Wl,--no-gc-sections %s 2>&1 | FileCheck -check-prefix=NO_GC_SECTIONS %s
// NO_GC_SECTIONS: clang{{.*}}" "-cc1" {{.*}} "-o" "[[temp:[^"]*]]"
// NO_GC_SECTIONS: lld{{.*}}" "-flavor" "ld" "--gc-sections" "-L/foo/lib32" "crt1.o" "crti.o" "--no-gc-sections" "[[temp]]" "-lc" "-lcompiler_rt" "crtn.o" "-o" "a.out"
