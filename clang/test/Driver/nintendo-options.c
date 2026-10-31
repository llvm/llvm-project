// REQUIRES: aarch64-registered-target

/// Check default compile and link options.
// RUN: %clang -### --target=aarch64-nintendo-nn %s 2>&1 | FileCheck -check-prefixes=DEFAULT %s
// DEFAULT: "{{.*}}clang{{.*}}" "-cc1"
// DEFAULT-SAME: "-triple" "aarch64-nintendo-nn"
// DEFAULT-SAME: "-target-cpu" "generic"
// DEFAULT-SAME: "-target-feature" "+v8a"
// DEFAULT-SAME: "-target-feature" "+fp-armv8"
// DEFAULT-SAME: "-target-feature" "+neon"
// DEFAULT-SAME: "-target-abi" "aapcs"

// DEFAULT: "{{.*}}ld.lld"
// DEFAULT-SAME: "-z" "relro" "-X"
// DEFAULT-SAME: "--build-id=sha1"
// DEFAULT-SAME: "--hash-style=gnu"
// DEFAULT-SAME: "--pack-dyn-relocs=relr"
// DEFAULT-SAME: "--eh-frame-hdr"

/// Check other compile and link options processed by the driver.
// RUN: %clang -### --target=aarch64-nintendo-nn -pie -rdynamic -s -flto -Wl,--no-demangle -g %s 2>&1 | FileCheck %s
// CHECK: "{{.*}}clang{{.*}}" "-cc1"
// CHECK-SAME: "-triple" "aarch64-nintendo-nn"
// CHECK-SAME: "-flto=full"
// CHECK-SAME: "-target-cpu" "generic"
// CHECK-SAME: "-target-feature" "+v8a"
// CHECK-SAME: "-target-feature" "+fp-armv8"
// CHECK-SAME: "-target-feature" "+neon"
// CHECK-SAME: "-target-abi" "aapcs"
// CHECK-SAME: "-dwarf-version=4"

// CHECK: "{{.*}}ld.lld"
// CHECK-SAME: "-pie"
// CHECK-SAME: "-export-dynamic" "-s"
// CHECK-SAME: "-z" "relro" "-X"
// CHECK-SAME: "--build-id=sha1"
// CHECK-SAME: "--hash-style=gnu"
// CHECK-SAME: "--pack-dyn-relocs=relr"
// CHECK-SAME: "--eh-frame-hdr"
// CHECK-SAME: "-plugin-opt=mcpu=generic"
// CHECK-SAME: "-plugin-opt=-enable-tlsdesc"
// CHECK-SAME: "--no-demangle"

/// Check assemble options not using integrated-as.
// RUN: %clang -### -c -x assembler --target=aarch64-nintendo-nn -gsplit-dwarf -fno-integrated-as %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=AS %s
// AS: "{{.*}}as" "-o"
// AS: "{{.*}}objcopy" "--extract-dwo"
// AS: "{{.*}}objcopy" "--strip-dwo"

/// Check invalid libcxx and rtlib option.
// RUN: %clangxx -### -x c++ --target=aarch64-nintendo-nn -stdlib=libc++ --rtlib=compiler-rt %s
// RUN: not %clangxx -### -x c++ --target=aarch64-nintendo-nn -stdlib=libstdc++ --rtlib=libgcc %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=STDRTLIBS %s
// STDRTLIBS: error: invalid library name in argument '-stdlib=libstdc++'
// STDRTLIBS: error: invalid runtime library name in argument '--rtlib=libgcc'

/// Check preprocessor option.
// RUN: %clang -### --target=aarch64-nintendo-nn -E %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=PP %s
// PP: "-E"
