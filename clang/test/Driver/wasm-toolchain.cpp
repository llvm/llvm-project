// A basic clang -cc1 command-line. WebAssembly is somewhat special in
// enabling -fvisibility=hidden by default.

// RUN: %clangxx -### %s --target=wasm32-unknown-unknown 2>&1 \
// RUN:   | FileCheck -check-prefix=CC1 %s
// CC1: "-cc1" "-triple" "wasm32-unknown-unknown" {{.*}} "-fvisibility=hidden" {{.*}}

// Ditto, but ensure that a user -fvisibility=default disables the default
// -fvisibility=hidden.

// RUN: %clangxx -### %s --target=wasm32-unknown-unknown -fvisibility=default 2>&1 \
// RUN:   | FileCheck -check-prefix=FVISIBILITY_DEFAULT %s
// FVISIBILITY_DEFAULT-NOT: hidden

// A basic C++ link command-line with unknown OS.

// RUN: %clangxx -### --target=wasm32-unknown-unknown --sysroot=/foo --stdlib=libc++ %s 2>&1 \
// RUN:   | FileCheck -check-prefix=LINK %s
// LINK: "-cc1" {{.*}} "-o" "[[temp:[^"]*]]"
// LINK: wasm-ld{{.*}}" "-L/foo/lib" "crt1.o" "[[temp]]" "-lc++" "-lc++abi" "-lc" "{{.*[/\\]}}libclang_rt.builtins.a" "-o" "a.out"

// RUN: %clangxx -### --target=wasm32-unknown-unknown --sysroot=/foo --stdlib=libstdc++ %s 2>&1 \
// RUN:   | FileCheck -check-prefix=LINK_STDCXX %s
// LINK_STDCXX: "-cc1" {{.*}} "-o" "[[temp:[^"]*]]"
// LINK_STDCXX: wasm-ld{{.*}}" "-L/foo/lib" "crt1.o" "[[temp]]" "-lstdc++" "-lc" "{{.*[/\\]}}libclang_rt.builtins.a" "-o" "a.out"

// A basic C++ link command-line with optimization with unknown OS.

// RUN: %clangxx -### -O2 --target=wasm32-unknown-unknown --sysroot=/foo %s --stdlib=libc++ 2>&1 \
// RUN:   | FileCheck -check-prefix=LINK_OPT %s
// LINK_OPT: "-cc1" {{.*}} "-o" "[[temp:[^"]*]]"
// LINK_OPT: wasm-ld{{.*}}" "-L/foo/lib" "crt1.o" "[[temp]]" "-lc++" "-lc++abi" "-lc" "{{.*[/\\]}}libclang_rt.builtins.a" "-o" "a.out"

// RUN: %clangxx -### -O2 --target=wasm32-unknown-unknown --sysroot=/foo %s --stdlib=libstdc++ 2>&1 \
// RUN:   | FileCheck -check-prefix=LINK_OPT_STDCXX %s
// LINK_OPT_STDCXX: "-cc1" {{.*}} "-o" "[[temp:[^"]*]]"
// LINK_OPT_STDCXX: wasm-ld{{.*}}" "-L/foo/lib" "crt1.o" "[[temp]]" "-lstdc++" "-lc" "{{.*[/\\]}}libclang_rt.builtins.a" "-o" "a.out"

// A basic C++ link command-line with known OS.

// RUN: %clangxx -### --target=wasm32-wasi --sysroot=/foo --stdlib=libc++ %s 2>&1 \
// RUN:   | FileCheck -check-prefix=LINK_KNOWN %s
// LINK_KNOWN: "-cc1" {{.*}} "-o" "[[temp:[^"]*]]"
// LINK_KNOWN: wasm-ld{{.*}}" "-L/foo/lib/wasm32-wasi" "crt1.o" "[[temp]]" "-lc++" "-lc++abi" "-lc" "{{.*[/\\]}}libclang_rt.builtins.a" "-o" "a.out"

// RUN: %clangxx -### --target=wasm32-wasi --sysroot=/foo --stdlib=libstdc++ %s 2>&1 \
// RUN:   | FileCheck -check-prefix=LINK_KNOWN_STDCXX %s
// LINK_KNOWN_STDCXX: "-cc1" {{.*}} "-o" "[[temp:[^"]*]]"
// LINK_KNOWN_STDCXX: wasm-ld{{.*}}" "-L/foo/lib/wasm32-wasi" "crt1.o" "[[temp]]" "-lstdc++" "-lc" "{{.*[/\\]}}libclang_rt.builtins.a" "-o" "a.out"

// A basic C++ link command-line with optimization with known OS.

// RUN: %clangxx -### -O2 --target=wasm32-wasi --sysroot=/foo %s --stdlib=libc++ 2>&1 \
// RUN:   | FileCheck -check-prefix=LINK_OPT_KNOWN %s
// LINK_OPT_KNOWN: "-cc1" {{.*}} "-o" "[[temp:[^"]*]]"
// LINK_OPT_KNOWN: wasm-ld{{.*}}" "-L/foo/lib/wasm32-wasi" "crt1.o" "[[temp]]" "-lc++" "-lc++abi" "-lc" "{{.*[/\\]}}libclang_rt.builtins.a" "-o" "a.out"

// RUN: %clangxx -### -O2 --target=wasm32-wasi --sysroot=/foo %s --stdlib=libstdc++ 2>&1 \
// RUN:   | FileCheck -check-prefix=LINK_OPT_KNOWN_STDCXX %s
// LINK_OPT_KNOWN_STDCXX: "-cc1" {{.*}} "-o" "[[temp:[^"]*]]"
// LINK_OPT_KNOWN_STDCXX: wasm-ld{{.*}}" "-L/foo/lib/wasm32-wasi" "crt1.o" "[[temp]]" "-lstdc++" "-lc" "{{.*[/\\]}}libclang_rt.builtins.a" "-o" "a.out"

// A basic C++ compile command-line with known OS.

// RUN: %clangxx -### --target=wasm32-wasi --stdlib=libc++ %s 2>&1 \
// RUN:     --sysroot=%S/Inputs/basic_linux_libcxx_tree/usr \
// RUN:   | FileCheck -check-prefix=COMPILE %s
// COMPILE: "-cc1"
// COMPILE: "-resource-dir" "[[RESOURCE_DIR:[^"]*]]"
// COMPILE: "-isysroot" "[[SYSROOT:[^"]+]]"
// COMPILE: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/wasm32-wasi/c++/v1"
// COMPILE: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/c++/v1"
// COMPILE: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
// COMPILE: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/wasm32-wasi"
// COMPILE: "-internal-isystem" "[[SYSROOT:[^"]+]]/include"

// RUN: %clangxx -### --target=wasm32-wasi --stdlib=libstdc++ %s 2>&1 \
// RUN:     --sysroot=%S/Inputs/basic_linux_libstdcxx_libcxxv2_tree/usr \
// RUN:   | FileCheck -check-prefix=COMPILE_STDCXX %s
// COMPILE_STDCXX: "-cc1"
// COMPILE_STDCXX: "-resource-dir" "[[RESOURCE_DIR:[^"]*]]"
// COMPILE_STDCXX: "-isysroot" "[[SYSROOT:[^"]+]]"
// COMPILE_STDCXX: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/c++/4.8/wasm32-wasi"
// COMPILE_STDCXX: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/c++/4.8"
// COMPILE_STDCXX: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/c++/4.8/backward"
// COMPILE_STDCXX: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
// COMPILE_STDCXX: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/wasm32-wasi"
// COMPILE_STDCXX: "-internal-isystem" "[[SYSROOT:[^"]+]]/include"

// RUN: %clangxx -### --target=wasm32-linux-muslwali --stdlib=libc++ %s 2>&1 \
// RUN:     --sysroot=%S/Inputs/basic_linux_libcxx_tree/usr \
// RUN:   | FileCheck -check-prefix=COMPILE_WALI %s
// COMPILE_WALI: "-cc1"
// COMPILE_WALI: "-resource-dir" "[[RESOURCE_DIR:[^"]*]]"
// COMPILE_WALI: "-isysroot" "[[SYSROOT:[^"]+]]"
// COMPILE_WALI: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/wasm32-linux-muslwali/c++/v1"
// COMPILE_WALI: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/c++/v1"
// COMPILE_WALI: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
// COMPILE_WALI: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/wasm32-linux-muslwali"
// COMPILE_WALI: "-internal-isystem" "[[SYSROOT:[^"]+]]/include"

// RUN: %clangxx -### --target=wasm32-linux-muslwali --stdlib=libstdc++ %s 2>&1 \
// RUN:     --sysroot=%S/Inputs/basic_linux_libstdcxx_libcxxv2_tree/usr \
// RUN:   | FileCheck -check-prefix=COMPILE_WALI_STDCXX %s
// COMPILE_WALI_STDCXX: "-cc1"
// COMPILE_WALI_STDCXX: "-resource-dir" "[[RESOURCE_DIR:[^"]*]]"
// COMPILE_WALI_STDCXX: "-isysroot" "[[SYSROOT:[^"]+]]"
// COMPILE_WALI_STDCXX: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/c++/4.8/wasm32-linux-muslwali"
// COMPILE_WALI_STDCXX: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/c++/4.8"
// COMPILE_WALI_STDCXX: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/c++/4.8/backward"
// COMPILE_WALI_STDCXX: "-internal-isystem" "[[RESOURCE_DIR]]{{(/|\\\\)}}include"
// COMPILE_WALI_STDCXX: "-internal-isystem" "[[SYSROOT:[^"]+]]/include/wasm32-linux-muslwali"
// COMPILE_WALI_STDCXX: "-internal-isystem" "[[SYSROOT:[^"]+]]/include"
