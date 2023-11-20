// A basic clang -cc1 command-line. WebAssembly is somewhat special in
// enabling -fvisibility=hidden by default.

// RUN: %clang -### %s --target=wasm32-unknown-unknown 2>&1 \
// RUN:   | FileCheck -check-prefix=CC1 %s
// CC1: "-cc1" "-triple" "wasm32-unknown-unknown" {{.*}} "-fvisibility=hidden" {{.*}}

// Ditto, but ensure that a user -fvisibility=default disables the default
// -fvisibility=hidden.

// RUN: %clang -### %s --target=wasm32-unknown-unknown -fvisibility=default 2>&1 \
// RUN:   | FileCheck -check-prefix=FVISIBILITY_DEFAULT %s
// FVISIBILITY_DEFAULT-NOT: hidden

// A basic C link command-line with unknown OS.

// RUN: %clang -### --target=wasm32-unknown-unknown --sysroot=/foo %s 2>&1 \
// RUN:   | FileCheck -check-prefix=LINK %s
// LINK: "-cc1" {{.*}} "-o" "[[temp:[^"]*]]"
// LINK: wasm-ld{{.*}}" "-L/foo/lib" "crt1.o" "[[temp]]" "-lc" "{{.*[/\\]}}libclang_rt.builtins-wasm32.a" "-o" "a.out"

// A basic C link command-line with optimization with unknown OS.

// RUN: %clang -### -O2 --target=wasm32-unknown-unknown --sysroot=/foo %s 2>&1 \
// RUN:   | FileCheck -check-prefix=LINK_OPT %s
// LINK_OPT: "-cc1" {{.*}} "-o" "[[temp:[^"]*]]"
// LINK_OPT: wasm-ld{{.*}}" "-L/foo/lib" "crt1.o" "[[temp]]" "-lc" "{{.*[/\\]}}libclang_rt.builtins-wasm32.a" "-o" "a.out"

// A basic C link command-line with known OS.

// RUN: %clang -### --target=wasm32-wasi --sysroot=/foo %s 2>&1 \
// RUN:   | FileCheck -check-prefix=LINK_KNOWN %s
// LINK_KNOWN: "-cc1" {{.*}} "-o" "[[temp:[^"]*]]"
// LINK_KNOWN: wasm-ld{{.*}}" "-L/foo/lib/wasm32-wasi" "crt1.o" "[[temp]]" "-lc" "{{.*[/\\]}}libclang_rt.builtins-wasm32.a" "-o" "a.out"

// -shared should be passed through to `wasm-ld` and include crt1-reactor.o with a known OS.

// RUN: %clang -### -shared -mexec-model=reactor --target=wasm32-wasi --sysroot=/foo %s 2>&1 \
// RUN:   | FileCheck -check-prefix=LINK_KNOWN_SHARED %s
// LINK_KNOWN_SHARED: "-cc1" {{.*}} "-o" "[[temp:[^"]*]]"
// LINK_KNOWN_SHARED: wasm-ld{{.*}}" "-L/foo/lib/wasm32-wasi" "crt1-reactor.o" "--entry" "_initialize" "-shared" "[[temp]]" "-lc" "{{.*[/\\]}}libclang_rt.builtins-wasm32.a" "-o" "a.out"

// -shared should be passed through to `wasm-ld` and include crt1-reactor.o with an unknown OS.

// RUN: %clang -### -shared -mexec-model=reactor --target=wasm32-unknown-unknown --sysroot=/foo %s 2>&1 \
// RUN:   | FileCheck -check-prefix=LINK_UNKNOWN_SHARED %s
// LINK_UNKNOWN_SHARED: "-cc1" {{.*}} "-o" "[[temp:[^"]*]]"
// LINK_UNKNOWN_SHARED: wasm-ld{{.*}}" "crt1-reactor.o" "--entry" "_initialize" "-shared" "[[temp]]" "-lc" "{{.*[/\\]}}libclang_rt.builtins-wasm32.a" "-o" "a.out"

// A basic C link command-line with optimization with known OS.

// RUN: %clang -### -O2 --target=wasm32-wasi --sysroot=/foo %s 2>&1 \
// RUN:   | FileCheck -check-prefix=LINK_OPT_KNOWN %s
// LINK_OPT_KNOWN: "-cc1" {{.*}} "-o" "[[temp:[^"]*]]"
// LINK_OPT_KNOWN: wasm-ld{{.*}}" "-L/foo/lib/wasm32-wasi" "crt1.o" "[[temp]]" "-lc" "{{.*[/\\]}}libclang_rt.builtins-wasm32.a" "-o" "a.out"

// A basic C compile command-line with known OS.

// RUN: %clang -### --target=wasm32-wasi --sysroot=/foo %s 2>&1 \
// RUN:   | FileCheck -check-prefix=COMPILE %s
// COMPILE: "-cc1" {{.*}} "-internal-isystem" "/foo/include/wasm32-wasi" "-internal-isystem" "/foo/include"

// -fPIC should work on a known OS

// RUN: %clang -### -fPIC --target=wasm32-wasi --sysroot=/foo %s 2>&1 \
// RUN:   | FileCheck -check-prefix=COMPILE_KNOWN_PIC %s
// COMPILE_KNOWN_PIC: "-cc1" {{.*}} "-mrelocation-model" "pic" "-pic-level" "2" {{.*}} "-internal-isystem" "/foo/include/wasm32-wasi" "-internal-isystem" "/foo/include"

// -fPIC should work on an unknown OS

// RUN: %clang -### -fPIC --target=wasm32-unknown-unknown --sysroot=/foo %s 2>&1 \
// RUN:   | FileCheck -check-prefix=COMPILE_UNKNOWN_PIC %s
// COMPILE_UNKNOWN_PIC: "-cc1" {{.*}} "-mrelocation-model" "pic" "-pic-level" "2"

// Thread-related command line tests.

// '-pthread' sets +atomics, +bulk-memory, +mutable-globals, +sign-ext, and --shared-memory
// RUN: %clang -### --target=wasm32-unknown-unknown \
// RUN:    --sysroot=/foo %s -fuse-ld=wasm-ld -pthread 2>&1 \
// RUN:  | FileCheck -check-prefix=PTHREAD %s
// PTHREAD: "-cc1" {{.*}} "-target-feature" "+atomics" "-target-feature" "+bulk-memory" "-target-feature" "+mutable-globals" "-target-feature" "+sign-ext"
// PTHREAD: wasm-ld{{.*}}" "-lpthread" "--shared-memory"

// '-pthread' not allowed with '-mno-atomics'
// RUN: %clang -### --target=wasm32-unknown-unknown \
// RUN:     --sysroot=/foo %s -pthread -mno-atomics 2>&1 \
// RUN:   | FileCheck -check-prefix=PTHREAD_NO_ATOMICS %s
// PTHREAD_NO_ATOMICS: invalid argument '-pthread' not allowed with '-mno-atomics'

// '-pthread' not allowed with '-mno-bulk-memory'
// RUN: %clang -### --target=wasm32-unknown-unknown \
// RUN:     --sysroot=/foo %s -pthread -mno-bulk-memory 2>&1 \
// RUN:   | FileCheck -check-prefix=PTHREAD_NO_BULK_MEM %s
// PTHREAD_NO_BULK_MEM: invalid argument '-pthread' not allowed with '-mno-bulk-memory'

// '-pthread' not allowed with '-mno-mutable-globals'
// RUN: %clang -### --target=wasm32-unknown-unknown \
// RUN:     --sysroot=/foo %s -pthread -mno-mutable-globals 2>&1 \
// RUN:   | FileCheck -check-prefix=PTHREAD_NO_MUT_GLOBALS %s
// PTHREAD_NO_MUT_GLOBALS: invalid argument '-pthread' not allowed with '-mno-mutable-globals'

// '-pthread' not allowed with '-mno-sign-ext'
// RUN: %clang -### --target=wasm32-unknown-unknown \
// RUN:     --sysroot=/foo %s -pthread -mno-sign-ext 2>&1 \
// RUN:   | FileCheck -check-prefix=PTHREAD_NO_SIGN_EXT %s
// PTHREAD_NO_SIGN_EXT: invalid argument '-pthread' not allowed with '-mno-sign-ext'

// '-mllvm -emscripten-cxx-exceptions-allowed=foo,bar' sets
// '-mllvm --force-attribute=foo:noinline -mllvm --force-attribute=bar:noinline'
// RUN: %clang -### --target=wasm32-unknown-unknown \
// RUN:    --sysroot=/foo %s -mllvm -enable-emscripten-cxx-exceptions \
// RUN:    -mllvm -emscripten-cxx-exceptions-allowed=foo,bar 2>&1 \
// RUN:  | FileCheck -check-prefix=EMSCRIPTEN_EH_ALLOWED_NOINLINE %s
// EMSCRIPTEN_EH_ALLOWED_NOINLINE: "-cc1" {{.*}} "-mllvm" "--force-attribute=foo:noinline" "-mllvm" "--force-attribute=bar:noinline"

// '-mllvm -emscripten-cxx-exceptions-allowed' only allowed with
// '-mllvm -enable-emscripten-cxx-exceptions'
// RUN: %clang -### --target=wasm32-unknown-unknown \
// RUN:     --sysroot=/foo %s -mllvm -emscripten-cxx-exceptions-allowed 2>&1 \
// RUN:   | FileCheck -check-prefix=EMSCRIPTEN_EH_ALLOWED_WO_ENABLE %s
// EMSCRIPTEN_EH_ALLOWED_WO_ENABLE: invalid argument '-mllvm -emscripten-cxx-exceptions-allowed' only allowed with '-mllvm -enable-emscripten-cxx-exceptions'

// '-fwasm-exceptions' sets +exception-handling and '-mllvm -wasm-enable-eh'
// RUN: %clang -### --target=wasm32-unknown-unknown \
// RUN:    --sysroot=/foo %s -fwasm-exceptions 2>&1 \
// RUN:  | FileCheck -check-prefix=WASM_EXCEPTIONS %s
// WASM_EXCEPTIONS: "-cc1" {{.*}} "-target-feature" "+exception-handling" "-mllvm" "-wasm-enable-eh"

// '-fwasm-exceptions' not allowed with '-mno-exception-handling'
// RUN: %clang -### --target=wasm32-unknown-unknown \
// RUN:     --sysroot=/foo %s -fwasm-exceptions -mno-exception-handling 2>&1 \
// RUN:   | FileCheck -check-prefix=WASM_EXCEPTIONS_NO_EH %s
// WASM_EXCEPTIONS_NO_EH: invalid argument '-fwasm-exceptions' not allowed with '-mno-exception-handling'

// '-fwasm-exceptions' not allowed with '-mllvm -enable-emscripten-cxx-exceptions'
// RUN: %clang -### --target=wasm32-unknown-unknown \
// RUN:     --sysroot=/foo %s -fwasm-exceptions \
// RUN:     -mllvm -enable-emscripten-cxx-exceptions 2>&1 \
// RUN:   | FileCheck -check-prefix=WASM_EXCEPTIONS_EMSCRIPTEN_EH %s
// WASM_EXCEPTIONS_EMSCRIPTEN_EH: invalid argument '-fwasm-exceptions' not allowed with '-mllvm -enable-emscripten-cxx-exceptions'

// '-mllvm -wasm-enable-sjlj' sets +exception-handling and
// '-exception-model=wasm'
// RUN: %clang -### --target=wasm32-unknown-unknown \
// RUN:    --sysroot=/foo %s -mllvm -wasm-enable-sjlj 2>&1 \
// RUN:  | FileCheck -check-prefix=WASM_SJLJ %s
// WASM_SJLJ: "-cc1" {{.*}} "-target-feature" "+exception-handling" "-exception-model=wasm"

// '-mllvm -wasm-enable-sjlj' not allowed with '-mno-exception-handling'
// RUN: %clang -### --target=wasm32-unknown-unknown \
// RUN:     --sysroot=/foo %s -mllvm -wasm-enable-sjlj -mno-exception-handling \
// RUN:     2>&1 \
// RUN:   | FileCheck -check-prefix=WASM_SJLJ_NO_EH %s
// WASM_SJLJ_NO_EH: invalid argument '-mllvm -wasm-enable-sjlj' not allowed with '-mno-exception-handling'

// '-mllvm -wasm-enable-sjlj' not allowed with
// '-mllvm -enable-emscripten-cxx-exceptions'
// RUN: %clang -### --target=wasm32-unknown-unknown \
// RUN:     --sysroot=/foo %s -mllvm -wasm-enable-sjlj \
// RUN:     -mllvm -enable-emscripten-cxx-exceptions 2>&1 \
// RUN:   | FileCheck -check-prefix=WASM_SJLJ_EMSCRIPTEN_EH %s
// WASM_SJLJ_EMSCRIPTEN_EH: invalid argument '-mllvm -wasm-enable-sjlj' not allowed with '-mllvm -enable-emscripten-cxx-exceptions'

// '-mllvm -wasm-enable-sjlj' not allowed with '-mllvm -enable-emscripten-sjlj'
// RUN: %clang -### --target=wasm32-unknown-unknown \
// RUN:     --sysroot=/foo %s -mllvm -wasm-enable-sjlj \
// RUN:     -mllvm -enable-emscripten-sjlj 2>&1 \
// RUN:   | FileCheck -check-prefix=WASM_SJLJ_EMSCRIPTEN_SJLJ %s
// WASM_SJLJ_EMSCRIPTEN_SJLJ: invalid argument '-mllvm -wasm-enable-sjlj' not allowed with '-mllvm -enable-emscripten-sjlj'

// RUN: %clang -### %s -fsanitize=address --target=wasm32-unknown-emscripten 2>&1 | FileCheck -check-prefix=CHECK-ASAN-EMSCRIPTEN %s
// CHECK-ASAN-EMSCRIPTEN: "-fsanitize=address"
// CHECK-ASAN-EMSCRIPTEN: "-fsanitize-address-globals-dead-stripping"

// RUN: %clang -### %s -fsanitize=function --target=wasm32-unknown-emscripten 2>&1 | FileCheck --check-prefix=FUNCTION %s
// FUNCTION: error: unsupported option '-fsanitize=function' for target 'wasm32-unknown-emscripten'

// Basic exec-model tests.

// RUN: %clang -### %s --target=wasm32-unknown-unknown --sysroot=%s/no-sysroot-there -mexec-model=command 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-COMMAND %s
// CHECK-COMMAND: "-cc1" {{.*}} "-o" "[[temp:[^"]*]]"
// CHECK-COMMAND: wasm-ld{{.*}}" "crt1.o" "[[temp]]" "-lc" "{{.*[/\\]}}libclang_rt.builtins-wasm32.a" "-o" "a.out"

// RUN: %clang -### %s --target=wasm32-unknown-unknown --sysroot=%s/no-sysroot-there -mexec-model=reactor 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-REACTOR %s
// CHECK-REACTOR: "-cc1" {{.*}} "-o" "[[temp:[^"]*]]"
// CHECK-REACTOR: wasm-ld{{.*}}" "crt1-reactor.o" "--entry" "_initialize" "[[temp]]" "-lc" "{{.*[/\\]}}libclang_rt.builtins-wasm32.a" "-o" "a.out"

// -fPIC implies +mutable-globals

// RUN: %clang -### %s --target=wasm32-unknown-unknown --sysroot=%s/no-sysroot-there -fPIC 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-PIC %s
// CHECK-PIC: "-cc1" {{.*}} "-target-feature" "+mutable-globals"

// '-mno-mutable-globals' is not allowed with '-fPIC'
// RUN: %clang -### %s --target=wasm32-unknown-unknown --sysroot=%s/no-sysroot-there -fPIC -mno-mutable-globals %s 2>&1 \
// RUN:   | FileCheck -check-prefix=PIC_NO_MUTABLE_GLOBALS %s
// PIC_NO_MUTABLE_GLOBALS: error: invalid argument '-fPIC' not allowed with '-mno-mutable-globals'
