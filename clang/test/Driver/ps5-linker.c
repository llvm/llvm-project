// Test that PIE is the default for main components

// RUN: %clang --target=x86_64-sie-ps5 %s -### 2>&1 | FileCheck --check-prefixes=CHECK-PIE %s

// CHECK-PIE: {{ld(\.exe)?}}"
// CHECK-PIE-SAME: "-pie"

// RUN: %clang --target=x86_64-sie-ps5 -no-pie %s -### 2>&1 | FileCheck --check-prefixes=CHECK-NO-PIE %s
// RUN: %clang --target=x86_64-sie-ps5 -r %s -### 2>&1 | FileCheck --check-prefixes=CHECK-NO-PIE %s
// RUN: %clang --target=x86_64-sie-ps5 -shared %s -### 2>&1 | FileCheck --check-prefixes=CHECK-NO-PIE,CHECK-SHARED %s
// RUN: %clang --target=x86_64-sie-ps5 -static %s -### 2>&1 | FileCheck --check-prefixes=CHECK-NO-PIE %s

// CHECK-NO-PIE: {{ld(\.exe)?}}"
// CHECK-NO-PIE-NOT: "-pie"
// CHECK-SHARED: "--shared"

// Test that -static is forwarded to the linker

// RUN: %clang --target=x86_64-sie-ps5 -static %s -### 2>&1 | FileCheck --check-prefixes=CHECK-STATIC %s

// CHECK-STATIC: {{ld(\.exe)?}}"
// CHECK-STATIC-SAME: "-static"

// Test the driver's control over the JustMyCode behavior with linker flags.

// RUN: %clang --target=x86_64-sie-ps5 -fjmc %s -### 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-LIB %s
// RUN: %clang --target=x86_64-sie-ps5 -flto -fjmc %s -### 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-LIB %s

// CHECK: -plugin-opt=-enable-jmc-instrument

// Check the default library name.
// CHECK-LIB: "--whole-archive" "-lSceJmc_nosubmission" "--no-whole-archive"

// Test the driver's control over the -fcrash-diagnostics-dir behavior with linker flags.

// RUN: %clang --target=x86_64-sie-ps5 -fcrash-diagnostics-dir=mydumps %s -### 2>&1 | FileCheck --check-prefixes=CHECK-DIAG %s
// RUN: %clang --target=x86_64-sie-ps5 -flto -fcrash-diagnostics-dir=mydumps %s -### 2>&1 | FileCheck --check-prefixes=CHECK-DIAG %s

// CHECK-DIAG: -plugin-opt=-crash-diagnostics-dir=mydumps

// Test the driver passes a sysroot to the linker. Without --sysroot, its value
// is sourced from the SDK environment variable.

// RUN: env SCE_PROSPERO_SDK_DIR=mysdk %clang --target=x64_64-sie-ps5 %s -### 2>&1 | FileCheck --check-prefixes=CHECK-SYSROOT %s
// RUN: env SCE_PROSPERO_SDK_DIR=other %clang --target=x64_64-sie-ps5 %s -### --sysroot=mysdk 2>&1 | FileCheck --check-prefixes=CHECK-SYSROOT %s

// CHECK-SYSROOT: {{ld(\.exe)?}}"
// CHECK-SYSROOT-SAME: "--sysroot=mysdk"

// Test that "." is always added to library search paths. This is long-standing
// behavior, unique to PlayStation toolchains.

// RUN: %clang --target=x64_64-sie-ps5 %s -### 2>&1 | FileCheck --check-prefixes=CHECK-LDOT %s

// CHECK-LDOT: {{ld(\.exe)?}}"
// CHECK-LDOT-SAME: "-L."

// Test that <sdk-root>/target/lib is added to library search paths, if it
// exists and no --sysroot is specified.

// RUN: rm -rf %t.dir && mkdir %t.dir
// RUN: env SCE_PROSPERO_SDK_DIR=%t.dir %clang --target=x64_64-sie-ps5 %s -### 2>&1 | FileCheck --check-prefixes=CHECK-NO-TARGETLIB %s
// RUN: env SCE_PROSPERO_SDK_DIR=%t.dir %clang --target=x64_64-sie-ps5 %s -### --sysroot=%t.dir 2>&1 | FileCheck --check-prefixes=CHECK-NO-TARGETLIB %s

// CHECK-NO-TARGETLIB: {{ld(\.exe)?}}"
// CHECK-NO-TARGETLIB-NOT: "-L{{.*[/\\]}}target/lib"

// RUN: mkdir -p %t.dir/target/lib
// RUN: env SCE_PROSPERO_SDK_DIR=%t.dir %clang --target=x64_64-sie-ps5 %s -### 2>&1 | FileCheck --check-prefixes=CHECK-TARGETLIB %s

// CHECK-TARGETLIB: {{ld(\.exe)?}}"
// CHECK-TARGETLIB-SAME: "-L{{.*[/\\]}}target/lib"
