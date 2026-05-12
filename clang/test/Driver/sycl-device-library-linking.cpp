// Test that SYCL device libraries are linked at compile-time for SPIR/SPIRV targets

// RUN: %clangxx -fsycl %s --sysroot=%S/Inputs/SYCL -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-DEVICE-LIBS

// RUN: %clangxx -fsycl --no-offloadlib %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-NO-DEVICE-LIBS

// Test Windows target includes libsycl-msvc-math.bc
// RUN: %clangxx --target=x86_64-pc-windows-msvc -fsycl %s --sysroot=%S/Inputs/SYCL -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-DEVICE-LIBS-WIN

// Test non-Windows target does not include libsycl-msvc-math.bc
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl %s --sysroot=%S/Inputs/SYCL -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-DEVICE-LIBS-LINUX

// Test ITT instrumentation libraries (enabled by default)
// RUN: %clangxx -fsycl %s --sysroot=%S/Inputs/SYCL -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-ITT-DEFAULT
// RUN: %clangxx -fsycl -fsycl-instrument-device-code %s --sysroot=%S/Inputs/SYCL -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-ITT-ENABLED
// RUN: %clangxx -fsycl -fno-sycl-instrument-device-code %s --sysroot=%S/Inputs/SYCL -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-ITT-DISABLED

// CHECK-DEVICE-LIBS: "-cc1" "-triple" "spirv64-unknown-unknown"
// CHECK-DEVICE-LIBS-SAME: "-fsycl-is-device"
// CHECK-DEVICE-LIBS-SAME: "-mlink-builtin-bitcode" "{{.*}}libsycl-crt.bc"
// CHECK-DEVICE-LIBS-SAME: "-mlink-builtin-bitcode" "{{.*}}libsycl-complex.bc"
// CHECK-DEVICE-LIBS-SAME: "-mlink-builtin-bitcode" "{{.*}}libsycl-complex-fp64.bc"
// CHECK-DEVICE-LIBS-SAME: "-mlink-builtin-bitcode" "{{.*}}libsycl-cmath.bc"
// CHECK-DEVICE-LIBS-SAME: "-mlink-builtin-bitcode" "{{.*}}libsycl-cmath-fp64.bc"
// CHECK-DEVICE-LIBS-SAME: "-mlink-builtin-bitcode" "{{.*}}libsycl-{{(msvc-math|imf)}}.bc"
// CHECK-DEVICE-LIBS-SAME: "-mlink-builtin-bitcode" "{{.*}}libsycl-imf-fp64.bc"
// CHECK-DEVICE-LIBS-SAME: "-mlink-builtin-bitcode" "{{.*}}libsycl-imf-bf16.bc"
// CHECK-DEVICE-LIBS-SAME: "-mlink-builtin-bitcode" "{{.*}}libsycl-fallback-cstring.bc"
// CHECK-DEVICE-LIBS-SAME: "-mlink-builtin-bitcode" "{{.*}}libsycl-fallback-complex.bc"
// CHECK-DEVICE-LIBS-SAME: "-mlink-builtin-bitcode" "{{.*}}libsycl-fallback-complex-fp64.bc"
// CHECK-DEVICE-LIBS-SAME: "-mlink-builtin-bitcode" "{{.*}}libsycl-fallback-cmath.bc"
// CHECK-DEVICE-LIBS-SAME: "-mlink-builtin-bitcode" "{{.*}}libsycl-fallback-cmath-fp64.bc"
// CHECK-DEVICE-LIBS-SAME: "-mlink-builtin-bitcode" "{{.*}}libsycl-fallback-imf.bc"
// CHECK-DEVICE-LIBS-SAME: "-mlink-builtin-bitcode" "{{.*}}libsycl-fallback-imf-fp64.bc"
// CHECK-DEVICE-LIBS-SAME: "-mlink-builtin-bitcode" "{{.*}}libsycl-fallback-imf-bf16.bc"
// CHECK-DEVICE-LIBS-SAME: "-mlink-builtin-bitcode-postopt"
// CHECK-DEVICE-LIBS-SAME: "-Wno-linker-warnings"

// CHECK-NO-DEVICE-LIBS: "-cc1" "-triple" "spirv64-unknown-unknown"
// CHECK-NO-DEVICE-LIBS-SAME: "-fsycl-is-device"
// CHECK-NO-DEVICE-LIBS-NOT: "-mlink-builtin-bitcode"
// CHECK-NO-DEVICE-LIBS-NOT: "libsycl-crt.bc"

// CHECK-ITT-DEFAULT: "-cc1" "-triple" "spirv64-unknown-unknown"
// CHECK-ITT-DEFAULT-SAME: "-fsycl-is-device"
// CHECK-ITT-DEFAULT-SAME: "-mlink-builtin-bitcode" "{{.*}}libsycl-itt-user-wrappers.bc"
// CHECK-ITT-DEFAULT-SAME: "-mlink-builtin-bitcode" "{{.*}}libsycl-itt-compiler-wrappers.bc"
// CHECK-ITT-DEFAULT-SAME: "-mlink-builtin-bitcode" "{{.*}}libsycl-itt-stubs.bc"

// CHECK-ITT-ENABLED: "-cc1" "-triple" "spirv64-unknown-unknown"
// CHECK-ITT-ENABLED-SAME: "-fsycl-is-device"
// CHECK-ITT-ENABLED-SAME: "-mlink-builtin-bitcode" "{{.*}}libsycl-itt-user-wrappers.bc"
// CHECK-ITT-ENABLED-SAME: "-mlink-builtin-bitcode" "{{.*}}libsycl-itt-compiler-wrappers.bc"
// CHECK-ITT-ENABLED-SAME: "-mlink-builtin-bitcode" "{{.*}}libsycl-itt-stubs.bc"

// CHECK-ITT-DISABLED: "-cc1" "-triple" "spirv64-unknown-unknown"
// CHECK-ITT-DISABLED-SAME: "-fsycl-is-device"
// CHECK-ITT-DISABLED-NOT: "libsycl-itt-user-wrappers.bc"
// CHECK-ITT-DISABLED-NOT: "libsycl-itt-compiler-wrappers.bc"
// CHECK-ITT-DISABLED-NOT: "libsycl-itt-stubs.bc"

// Windows target should include libsycl-msvc-math.bc
// CHECK-DEVICE-LIBS-WIN: "-cc1" "-triple" "spirv64-unknown-unknown"
// CHECK-DEVICE-LIBS-WIN-SAME: "-aux-triple" "x86_64-pc-windows-msvc"
// CHECK-DEVICE-LIBS-WIN-SAME: "-fsycl-is-device"
// CHECK-DEVICE-LIBS-WIN-SAME: "-mlink-builtin-bitcode" "{{.*}}libsycl-cmath-fp64.bc"
// CHECK-DEVICE-LIBS-WIN-SAME: "-mlink-builtin-bitcode" "{{.*}}libsycl-msvc-math.bc"
// CHECK-DEVICE-LIBS-WIN-SAME: "-mlink-builtin-bitcode" "{{.*}}libsycl-imf.bc"

// Linux target should NOT include libsycl-msvc-math.bc
// CHECK-DEVICE-LIBS-LINUX: "-cc1" "-triple" "spirv64-unknown-unknown"
// CHECK-DEVICE-LIBS-LINUX-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"
// CHECK-DEVICE-LIBS-LINUX-SAME: "-fsycl-is-device"
// CHECK-DEVICE-LIBS-LINUX-SAME: "-mlink-builtin-bitcode" "{{.*}}libsycl-cmath-fp64.bc"
// CHECK-DEVICE-LIBS-LINUX-NOT: "libsycl-msvc-math.bc"
// CHECK-DEVICE-LIBS-LINUX-SAME: "-mlink-builtin-bitcode" "{{.*}}libsycl-imf.bc"

void foo() {}
